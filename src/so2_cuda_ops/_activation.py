"""Activation-space SO2 pack, grouped GEMM and output scatter."""
import torch
import torch.nn.functional as F
from ._permutation import permute_rows

def _entry_map(bases, levels, dim, device):
    """For every feature f < dim, the (block, channel, d, l) entries of the irrep
    channels that cover f, as the output-major scatter kernels take them:
    (entry_offsets[dim + 1], entry_m, entry_channel, entry_d, entry_l)."""
    per_feature = [[] for _ in range(dim)]
    for block, (base_t, l_t) in enumerate(zip(bases, levels)):
        for channel, (base, l) in enumerate(zip(base_t.tolist(), l_t.tolist())):
            for d in range(2 * l + 1):
                per_feature[base + d].append((block, channel, d, l))
    offsets = [0]
    for entries in per_feature:
        offsets.append(offsets[-1] + len(entries))
    columns = list(zip(*(e for entries in per_feature for e in entries))) or [(), (), (), ()]
    as_long = lambda values: torch.tensor(list(values), dtype=torch.long, device=device)  # noqa: E731
    return (as_long(offsets),) + tuple(as_long(column) for column in columns)


def _layer_layout(module, device, in_dim):
    """Pair maps of every m block and the multi-m layout of the m>0 blocks, cached on the layer.

    The pack and scatter functions save these tensors for backward, so they are built
    outside inference mode (an inference-mode warm-up would otherwise leave inference
    tensors for a later training call), with their own pair-map cache on the layer."""
    cache = getattr(module, "_so2_activation_layout", None)
    if cache is None:
        cache = {}
        module._so2_activation_layout = cache
    key = (str(device), int(in_dim))
    hit = cache.get(key)
    if hit is not None:
        return hit
    from so2_cuda_ops.so2_sandwich_common import so2_pair_maps

    with torch.inference_mode(False), torch.no_grad():
        maps = [so2_pair_maps(module, m, device, cache_attr="_so2_activation_pair_maps")
                for m in range(module.m_max + 1)]
        hit = {"m0": maps[0], "multi": None}
        if module.m_max >= 1:
            maps = maps[1:]
            in_bases = [p[0] for p in maps]
            in_ls = [p[1] for p in maps]
            out_bases = [p[2] for p in maps]
            out_ls = [p[3] for p in maps]
            cins = [int(b.numel()) for b in in_bases]

            def prefix(sizes):
                values = [0]
                for size in sizes:
                    values.append(values[-1] + size)
                return torch.tensor(values, dtype=torch.long, device=device)

            hit["multi"] = {
                "in_bases": in_bases,
                "in_ls": in_ls,
                "out_bases": out_bases,
                "out_ls": out_ls,
                "cins": cins,
                "cin_prefix": prefix(cins),
                "cout_prefix": prefix([int(b.numel()) for b in out_bases]),
                "m_values": torch.tensor(list(range(1, module.m_max + 1)), dtype=torch.long, device=device),
                "in_entries": _entry_map(in_bases, in_ls, int(in_dim), device),
                "out_entries": _entry_map(out_bases, out_ls, int(module.irreps_out.dim), device),
            }
    cache[key] = hit
    return hit


class _PackAll(torch.autograd.Function):
    """SO2CUDA packing of a layer input: the m0 block [n, cin0] and, by the multi-m
    pack, every m>0 block as pairs [n, 2, sum cin_m].

    Packing reads each packed value from its irrep block through the rotation, so the
    backward is the transposed rotation scattered into the input layout.  It is written
    output-major (the m0 scatter and the multi-m pair scatter through the input maps):
    one thread sums each input-gradient element, where the packs' own backwards add
    every block with atomics, several m blocks onto the same element."""

    @staticmethod
    def forward(ctx, x, ops, wigner, compact_offsets, mode, stride, rotate, m0_maps, lay):
        in_base, in_l, offsets = m0_maps
        inp0 = ops._pack_m0_cuda(x, wigner, in_base, in_l, offsets, compact_offsets, rotate, mode, stride)
        pairs = ops._pack_pairs_multi_cuda(
            x, wigner, lay["in_bases"], lay["in_ls"], offsets, compact_offsets,
            lay["cin_prefix"], lay["m_values"], rotate, mode, stride)
        ctx.save_for_backward(wigner)
        ctx.meta = (ops, compact_offsets, mode, stride, rotate, m0_maps, lay, int(x.shape[1]))
        return inp0, pairs

    @staticmethod
    def backward(ctx, grad_inp0, grad_pairs):
        (wigner,) = ctx.saved_tensors
        ops, compact_offsets, mode, stride, rotate, (in_base, in_l, offsets), lay, in_dim = ctx.meta
        grad_x = None
        if grad_inp0 is not None:
            grad_x = ops._scatter_m0_forward_cuda(
                grad_inp0.contiguous(), wigner, in_base, in_l, offsets, compact_offsets,
                in_dim, rotate, mode, stride)
        if grad_pairs is not None:
            blocks = [block.contiguous() for block in torch.split(grad_pairs, lay["cins"], dim=-1)]
            part = ops._scatter_pairs_multi_output_major_forward_cuda(
                blocks, wigner, offsets, compact_offsets, lay["cin_prefix"], lay["m_values"],
                *lay["in_entries"], in_dim, rotate, mode, stride)
            grad_x = part if grad_x is None else grad_x + part
        return grad_x, None, None, None, None, None, None, None, None


class _Packed:
    """The SO2CUDA packing of one layer call.

    ``_PackAll`` packs m0 and, in one multi-m pack, every m>0 block; its backward writes
    each input-gradient element once.  ``scatter`` writes the rotated output of all
    blocks in one output-major pass instead of a full-width output per block."""

    def __init__(self, ops, module, x, wigner_info, radials):
        self.ops = ops
        self.module = module
        self.wigner, self.compact_offsets, self.mode, self.stride = wigner_info
        x = x.contiguous()
        layout = _layer_layout(module, x.device, x.shape[1])
        ib, il, self.m0_out_base, self.m0_out_l, self.offsets = layout["m0"]
        rotate = bool(module.rotate_in)
        self.layout = layout["multi"]
        if self.layout is None:
            inp0 = ops._PackM0Function.apply(
                x, self.wigner, ib, il, self.offsets, self.compact_offsets, rotate, self.mode, self.stride)
            packed = None
        else:
            inp0, packed = _PackAll.apply(x, ops, self.wigner, self.compact_offsets, self.mode, self.stride,
                                          rotate, (ib, il, self.offsets), self.layout)
        front = radials is not None and module.front
        if front:
            inp0 = inp0 * radials[0]
        self.inputs = [inp0]
        if packed is not None:
            if front:
                packed = packed * radials[1].unsqueeze(1)
            # [n, 2, cin_m] views; each reshapes to [2n, cin_m] rows without a copy
            self.inputs.extend(torch.split(packed, self.layout["cins"], dim=-1))
        self.radials = None if radials is None or module.front else radials

    def finish_m0(self, y):
        """The m0 output with the radial weight of a layer that scales its outputs."""
        return y if self.radials is None else y * self.radials[0]

    def finish_raw(self, m, raw):
        """Raw pair output [n, 2, 2C] of block m with the output radial folded in:
        complex_pair_output(raw) * r equals complex_pair_output(raw * [r, r])."""
        if self.radials is None:
            return raw
        radial = self.radials[m]
        return raw * torch.cat((radial, radial), dim=-1).unsqueeze(1)

    def scatter(self, y0, raws):
        """Rotated layer output from the finished m0 block and the raw m>0 blocks."""
        module = self.module
        head = (y0.contiguous(), self.wigner, self.m0_out_base, self.m0_out_l, self.offsets, self.compact_offsets)
        rotate = (module.rotate_out, self.mode, self.stride)
        if self.layout is None:
            return self.ops._ScatterM0OutputFunction.apply(*head, module.irreps_out.dim, *rotate)
        lay = self.layout
        return self.ops._ScatterM0RawPairsMultiOutputMajorFunction.apply(
            *head, lay["cout_prefix"], lay["m_values"], *lay["out_entries"], module.irreps_out.dim, *rotate,
            len(raws), *(raw.contiguous() for raw in raws), *lay["out_bases"], *lay["out_ls"])


def _pair_rows(index):
    """Row index over [n] -> row index over the [n, 2] real/imaginary pair rows."""
    two = torch.arange(2, device=index.device, dtype=index.dtype)
    return (index.unsqueeze(1) * 2 + two).reshape(-1)


@torch.inference_mode(False)
@torch.no_grad()
def _slot_layouts(mole_globals, idx, num_experts, schedule):
    """Expert-sorted row layouts of the grouped GEMM, cached on mole_globals (every SO2
    layer of one forward shares the routing).

    'per_slot' has one layout per slot j, the expert sort of MOLEGlobals.expert_slot_layout
    shared with MOLELinear and top1_prior.linear; 'expanded' one layout over the n*k rows
    r = e*k + j of the flattened top-k table.  The cache follows the storage and version
    of the routing tensor; inference tensors, without a version counter, are not cached."""
    try:
        version = int(idx._version)
    except RuntimeError:
        version = None
    source = getattr(mole_globals, "_activation_fused_p0_source", None)
    cache = getattr(mole_globals, "_activation_fused_p0_layouts", None)
    if (cache is None or version is None or source is None
            or source[0] is not idx or source[1] != version):
        cache = {}
        mole_globals._activation_fused_p0_layouts = cache
        mole_globals._activation_fused_p0_source = (idx, version)
    key = (schedule, str(idx.device), tuple(idx.shape), int(num_experts))
    hit = cache.get(key)
    if hit is not None:
        return hit
    n, k = idx.shape
    slots = [(idx.reshape(-1), k)] if schedule == "expanded" else [(idx[:, j], 1) for j in range(k)]
    hit = []
    for slot, (flat, width) in enumerate(slots):
        flat = flat.to(torch.long)
        if schedule == "per_slot":
            order, inverse, ptr, _ = mole_globals.expert_slot_layout(slot, flat, num_experts)
        else:
            order = torch.argsort(flat, stable=True)
            inverse = torch.empty_like(order)
            inverse.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))
            ptr = torch.zeros(int(num_experts) + 1, dtype=torch.long, device="cpu")
            ptr[1:] = torch.cumsum(torch.bincount(flat, minlength=int(num_experts)).cpu(), dim=0)
        # 'expanded' gathers each edge row once per slot (order // k); only the
        # per-slot gathers and the inverse gathers are permutations.
        gather = torch.div(order, width, rounding_mode="floor") if width > 1 else order
        hit.append({
            "gather": gather,
            "order": order,
            "inverse": inverse,
            "ptr": ptr,
            "pair_gather": _pair_rows(gather),
            "pair_order": _pair_rows(order),
            "pair_inverse": _pair_rows(inverse),
            "pair_ptr": ptr * 2,
        })
    if version is not None:
        cache[key] = hit
    return hit


def _fused_p0(module, x, packed, mole_globals, linears, schedule, grouped_gemm_multi):
    idx = mole_globals.topk_indices.to(device=x.device, dtype=torch.long)
    val = mole_globals.topk_values.to(device=x.device, dtype=x.dtype)
    n, k = idx.shape
    branch = getattr(mole_globals, "branch", "all")
    fold = bool(getattr(mole_globals, "coefficients_sum_to_one", False)) and branch == "all"
    routed = [m for m, (fc, _, _) in enumerate(linears) if fc is not None]
    # an expert bias is one more weight column against a column of ones: the GEMM adds
    # it and its gradient is the per-expert row sum of the GEMM's weight gradient
    weights = [linears[m][1] if linears[m][2] is None
               else torch.cat((linears[m][1], linears[m][2].unsqueeze(-1)), dim=-1) for m in routed]
    mixed = {}
    slot_layouts = [] if branch == "shared" else _slot_layouts(mole_globals, idx, module.fc_m0.num_experts, schedule)
    for j, lay in enumerate(slot_layouts):
        xs, ptrs = [], []
        for m in routed:
            flat = packed.inputs[m].reshape(-1, packed.inputs[m].shape[-1])
            pre = "" if m == 0 else "pair_"
            if schedule == "per_slot":
                rows = permute_rows(flat, lay[pre + "order"], lay[pre + "inverse"])
            else:
                rows = flat.index_select(0, lay[pre + "gather"])
            if linears[m][2] is not None:
                rows = torch.cat((rows, rows.new_ones(rows.shape[0], 1)), dim=1)
            xs.append(rows)
            ptrs.append(lay[pre + "ptr"])
        ys = grouped_gemm_multi(xs, ptrs, weights) if routed else []
        for m, y in zip(routed, ys):
            pre = "" if m == 0 else "pair_"
            y = permute_rows(y, lay[pre + "inverse"], lay[pre + "order"])
            inp = packed.inputs[m]
            if schedule == "expanded":
                # rows in (edge, slot[, pair]) order: the sum over slots of
                # MOLELinear._apply_activation_space
                y = y.reshape(n, k, *inp.shape[1:-1], y.shape[-1])
                view = [n] + [1] * (y.dim() - 2)
                parts = [y[:, s] * val[:, s].reshape(view) for s in range(k)]
            else:
                y = y.reshape(*inp.shape[:-1], y.shape[-1])
                parts = [y * val[:, j].reshape([n] + [1] * (y.dim() - 1))]
            for part in parts:
                mixed[m] = part if m not in mixed else mixed[m] + part

    raws = []
    for m, (fc, _, _) in enumerate(linears):
        inp = packed.inputs[m]
        if fc is None:
            if branch == "routed":  # a non-MoLE block belongs to the shared branch
                raw = inp.new_zeros(*inp.shape[:-1], 2 * module.m_linear[m - 1].num_out_channel)
            else:
                raw = module.m_linear[m - 1].fc(inp)  # interpolation block
        else:
            raw = mixed.get(m)
            if raw is None:
                raw = inp.new_zeros(*inp.shape[:-1], fc.out_features)
            if branch != "routed" and not fold and fc.num_shared_experts > 0:
                shared_bias = fc.bias_shared.sum(0) if fc.bias_shared is not None else None
                raw = raw + F.linear(inp, fc.weight_shared.sum(0), shared_bias)
        raws.append(packed.finish_m0(raw) if m == 0 else packed.finish_raw(m, raw))
    return packed.scatter(raws[0], raws[1:])


