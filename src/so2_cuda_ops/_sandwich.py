"""Block-layout SO(2) sandwich: rotate into m blocks, one GEMM per block, rotate back.

The data of one call live in one flat buffer with one block per m. Block m > 0
holds one row per edge, ``[x_{-m} (C_m) | x_{+m} (C_m)]``, so the SO(2) linear of
the block is a single GEMM against the block-complex weight ``[[A, -B], [B, A]]``
built from the stacked pair weight ``[A; B]``:

    [y_{-m} | y_{+m}] = [x_{-m} | x_{+m}] @ [[A, -B], [B, A]]^T

Block 0 (optional) holds the m = 0 coefficients and uses the plain [Cout, Cin]
weight and bias. Rotation into the blocks and back out of them are channel-major
kernels: one thread owns one (edge, irrep channel), keeps its 2l+1 coefficients
in registers and reads or writes every coefficient of the feature row once.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

MAX_BLOCKS = 16


def _ext():
    from .tensor_product import _load_extension

    return _load_extension()


def _side(all_bases, all_ls, block_members, dim, device):
    """Channel tables of one side (input or output) of a layer.

    ``block_members[m]`` lists the channels (first features) of block m in column
    order, or is None when block m is absent. ``cols[k, m]`` is the column of
    channel k in block m, -1 when the channel is not in that block."""
    bases = [int(b) for b in all_bases.detach().cpu().tolist()]
    levels = [int(v) for v in all_ls.detach().cpu().tolist()]
    index = {b: k for k, b in enumerate(bases)}
    width = len(block_members)
    cols = [[-1] * width for _ in bases]
    widths, strides, prefix = [], [], []
    total = 0
    for m, members in enumerate(block_members):
        prefix.append(total)
        if members is None:
            widths.append(0)
            strides.append(0)
            continue
        members = [int(b) for b in members.detach().cpu().tolist()]
        for c, b in enumerate(members):
            cols[index[b]][m] = c
        widths.append(len(members))
        strides.append(len(members) if m == 0 else 2 * len(members))
        total += strides[-1]
    covered = [0] * int(dim)
    for b, l in zip(bases, levels):
        for f in range(b, b + 2 * l + 1):
            if 0 <= f < int(dim):
                covered[f] += 1

    def as_int(values):
        return torch.tensor(values, dtype=torch.int32, device=device).contiguous()

    table = [v for m in range(width) for v in (prefix[m], widths[m], strides[m])]
    radial_offsets = [sum(widths[:m]) for m in range(width)]
    return SimpleNamespace(
        base=as_int(bases), l=as_int(levels), cols=as_int([c for row in cols for c in row]),
        table=torch.tensor(table, dtype=torch.long, device=device),
        prefix=prefix, width=widths, stride=strides, total=total, radial_offsets=radial_offsets,
        zero_fill=any(c != 1 for c in covered), dim=int(dim))


def sandwich_plan(layout, in_dim, device, *, with_m0=False):
    """Blocks and channel tables of a layout, cached on the layout object."""
    cache = layout.__dict__.setdefault("_sandwich_plans", {})
    key = (int(in_dim), bool(with_m0), str(device))
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Load (on first use, build) the extension before any table reaches the device.
    # Loading it right after the tables' host-to-device copies, in a process that
    # had run a CUDA profiling tool, crashed the first kernel launch in the driver.
    _ext()
    maps = layout.maps
    in_members, out_members, blocks = [], [], []
    for m, (in_base, _in_l, out_base, _out_l, _offsets) in enumerate(maps):
        present = in_base.numel() > 0 and out_base.numel() > 0 and (m > 0 or with_m0)
        in_members.append(in_base if present else None)
        out_members.append(out_base if present else None)
        if present:
            blocks.append((m, int(in_base.numel()), int(out_base.numel())))
    with torch.inference_mode(False), torch.no_grad():
        hit = SimpleNamespace(
            supported=len(maps) <= MAX_BLOCKS,  # the kernels address at most 16 m blocks
            blocks=tuple(blocks),
            inp=_side(maps[0][0], maps[0][1], in_members, in_dim, device),
            out=_side(maps[0][2], maps[0][3], out_members, layout.out_dim, device),
            offsets=maps[0][-1],
            no_rows=torch.empty(0, dtype=torch.long, device=device),
            no_scale=torch.empty(0, dtype=torch.float32, device=device),
        )
    cache[key] = hit
    return hit


def _rotate(src, side, plan, wigner, rotate, scale=None, rows=None, copies=1, radials=(), plain=False):
    """Rotate ``src`` into the blocks of ``copies`` buffers laid out one after another:
    copy c stores edge e in row ``rows[c * N + e]``, scaled by ``scale[c * N + e]`` and, per
    block m and column, by ``radials[m]``. With ``plain`` one more buffer follows with the
    unscaled values in edge order."""
    return _ext().channel_rotate_to_blocks_fp32(
        src, wigner.values, plan.offsets, wigner.compact_offsets, side.base, side.l, side.cols,
        side.table, side.total, plan.no_scale if scale is None else scale,
        plan.no_rows if rows is None else rows, bool(rotate), int(wigner.mode), int(wigner.stride), int(copies),
        list(radials), bool(plain))


def _gather(src, n, side, plan, wigner, rotate, scale=None, rows=None, into=None, copies=1, radials=(),
            plain=None, radial_grad=None, dot_src=None, dot_out=None):
    """Rotate back sum_c scale_c * (block rows of copy c) into feature rows [N, dim].

    With ``radials`` the sum is first scaled per block column by the radial weights and,
    when ``radial_grad`` is given, the radial gradients are written from ``plain`` (the
    unscaled rotated input). ``dot_out[c * N + e]`` receives the dot product of edge e's
    rows of copy c in ``src`` and ``dot_src``."""
    return _ext().channel_gather_from_blocks_fp32(
        src, int(n), wigner.values, plan.offsets, wigner.compact_offsets, side.base, side.l, side.cols,
        side.table, side.dim, bool(side.zero_fill), plan.no_scale if scale is None else scale,
        plan.no_scale if into is None else into, plan.no_rows if rows is None else rows, bool(rotate),
        int(wigner.mode), int(wigner.stride), int(copies), list(radials),
        plan.no_scale if plain is None else plain, plan.no_scale if radial_grad is None else radial_grad,
        list(side.radial_offsets), plan.no_scale if dot_src is None else dot_src,
        plan.no_scale if dot_out is None else dot_out)


_SHARED_MEMORY = {}


def _edge_tiles_fit(plan, wigner, in_dim, device):
    """Whether the edge-tiled kernels can stage the input side of this layer (they carry the
    fused radial weights and gate products)."""
    mode = int(wigner.mode)
    if mode not in (0, 2):
        return False
    key = str(device)
    if key not in _SHARED_MEMORY:
        props = torch.cuda.get_device_properties(device)
        _SHARED_MEMORY[key] = int(getattr(props, "shared_memory_per_block_optin", 48 * 1024))
    wigner_floats = int(wigner.stride) if mode == 2 else 0
    need = max(in_dim + wigner_floats, plan.inp.total + wigner_floats + in_dim) * 4
    return need <= _SHARED_MEMORY[key]


def _radials_by_m(plan, radials):
    """Radial weight of every input block indexed by m (empty for blocks without one)."""
    by_m = [plan.no_scale] * len(plan.inp.width)
    for b, (m, _cin, _cout) in enumerate(plan.blocks):
        by_m[m] = radials[b]
    return by_m


def _copy_rows(slots, n, device):
    """[K * N] block row of every edge in every slot, or None when every slot keeps the edge order."""
    if all(slot.rows is None for slot in slots):
        return None
    rows = [slot.rows if slot.rows is not None else torch.arange(n, device=device) for slot in slots]
    return rows[0].contiguous() if len(rows) == 1 else torch.cat(rows)


def _gemm_ext():
    from ._cublas_grouped_gemm import _load_extension

    return _load_extension()


def _gemm_many(xs, weights, outs, ptr, transpose=False):
    """outs[p] = xs[p] @ W_p^T (@ W_p when transpose); with ``ptr`` the rows ptr[g]:ptr[g+1]
    of every problem use W_p[g] and all problems go to one cuBLAS grouped call."""
    if not xs:
        return
    if ptr is None:
        for x, weight, out in zip(xs, weights, outs):
            torch.mm(x, weight if transpose else weight.t(), out=out)
        return
    from ._cublas_grouped_gemm import _loop_max
    _gemm_ext().grouped_gemm_into_fp32(list(xs), ptr, list(weights), list(outs), bool(transpose), False,
                                       _loop_max(tuple(weights)))


def _weight_grad_many(grads, xs, outs, ptr, accumulate):
    """outs[p] (+)= grads[p]^T @ xs[p], per row segment of ``ptr`` when given."""
    if not xs:
        return
    if ptr is None:
        for grad, x, out in zip(grads, xs, outs):
            if accumulate:
                out.addmm_(grad.t(), x)
            else:
                torch.mm(grad.t(), x, out=out)
        return
    from ._cublas_grouped_gemm import _loop_max
    _gemm_ext().grouped_gemm_weight_grad_into_fp32(list(grads), list(xs), ptr, list(outs), bool(accumulate),
                                                   False, _loop_max(tuple(outs)))


def _pair_shape(spec, groups, cout, cin):
    return (groups, 2 * cout, 2 * cin) if spec.groups else (2 * cout, 2 * cin)


def _view(flat, n, prefix, width):
    return flat[n * prefix:n * (prefix + width)].view(n, width)


def _scale_block(block, radial, pairs):
    """block [N, w] times radial [N, c] per channel (both pair halves when pairs)."""
    n = block.shape[0]
    if pairs:
        return (block.view(n, 2, -1) * radial.unsqueeze(1)).view(n, -1)
    return block * radial


def _rowdot(a, b, pairs, n):
    """Per-channel product of a and b summed over the pair halves: [N, c]."""
    if pairs:
        return (a.view(n, 2, -1) * b.view(n, 2, -1)).sum(dim=1)
    return a * b


def _rows_of(slot, n):
    """Group of every block row of a grouped slot, as a device index [N] (computed once per slot).

    Callers that already hold the sorted group ids pass them to ``_slot``; otherwise they
    are expanded from the CPU pointer, which costs one host-to-device copy."""
    if getattr(slot, "group_rows", None) is None:
        counts = (slot.ptr[1:] - slot.ptr[:-1]).to(slot.device)
        slot.group_rows = torch.repeat_interleave(torch.arange(counts.numel(), device=slot.device), counts,
                                                  output_size=n)
    return slot.group_rows


class _Sandwich(torch.autograd.Function):
    """Rotate in, one GEMM per block, rotate out, with the backward written out.

    ``spec`` (static) describes the trailing tensors, in order: the m = 0 weight and
    bias when present, one stacked pair weight per m > 0 block, one radial weight per
    block (front: [N, Cin_b] scales the block's input before its GEMM; back:
    [N, Cout_b] scales its output) and one per-edge gate [N] per slot.

    The result is the sum over ``spec.slots``. Each slot stores edge e in block row
    ``slot.rows[e]`` (``slot.order`` is the inverse; both None for the edge order) of
    its own copy of the block buffer; one rotation writes every copy and one gather
    reads them all. When the weights are grouped (``spec.groups`` = G > 0: weights
    [G, Cout_0, Cin_0], bias [G, Cout_0], pair weights [G, 2*Cout_m, Cin_m]), the block
    rows ``slot.ptr[g]:slot.ptr[g+1]`` use weight g. A gated slot is scaled per edge
    by its gate."""

    @staticmethod
    def forward(ctx, x, plan, layout, wigner, spec, *tensors):
        ext = _ext()
        n = x.shape[0]
        inp, out = plan.inp, plan.out
        t = list(tensors)
        w0 = t.pop(0) if spec.has_m0 else None
        b0 = t.pop(0) if spec.has_bias else None
        pair_weights = [t.pop(0) for _ in range(spec.n_pairs)]
        radials = [t.pop(0) for _ in range(spec.n_radials)]
        gates = [t.pop(0) for _ in range(spec.n_gates)]
        groups = max(spec.groups, 1)
        needs = ctx.needs_input_grad
        idx = 5
        grad_w0 = bool(spec.has_m0 and needs[idx])
        idx += int(spec.has_m0)
        grad_b0 = bool(spec.has_bias and needs[idx])
        idx += int(spec.has_bias)
        grad_pairs = any(needs[idx:idx + spec.n_pairs])
        idx += spec.n_pairs
        grad_radials = any(needs[idx:idx + spec.n_radials])
        idx += spec.n_radials
        grad_gates = any(needs[idx:idx + spec.n_gates])
        need_x = needs[0]
        front_grad = spec.radial_mode == "front" and grad_radials
        back_grad = spec.radial_mode == "back" and grad_radials
        fused = spec.fused_radial
        if fused:
            # Front radial weights are applied by the rotation kernel. Only the input is kept;
            # the backward rotates it again, and the gate gradients come from the gather.
            keep_packed = keep_blocks = False
            keep_input = grad_w0 or grad_pairs or front_grad or grad_gates
            keep_weights = need_x or front_grad or grad_gates
        else:
            keep_input = False
            keep_packed = grad_w0 or grad_pairs or front_grad
            keep_weights = need_x or front_grad
            keep_blocks = back_grad or grad_gates
        block_weights = (ext.block_complex_weights_fp32([w.contiguous() for w in pair_weights])
                         if pair_weights else None)
        slots = spec.slots
        copies = len(slots)
        rows = _copy_rows(slots, n, x.device)
        gate_scale = torch.stack(gates).reshape(-1).contiguous() if gates else None
        in_size, out_size = n * inp.total, n * out.total
        # One rotation writes the block rows of every slot; one gather below reads them all.
        packed_all = _rotate(x, inp, plan, wigner, layout.rotate_in, rows=rows, copies=copies,
                             radials=_radials_by_m(plan, radials) if fused else ())
        blocks_all = x.new_empty(copies * out_size)
        scaled_all = blocks_all if spec.radial_mode != "back" else torch.empty_like(blocks_all)
        slot_radials_all = []
        for j, slot in enumerate(slots):
            packed = packed_all[j * in_size:(j + 1) * in_size]
            blocks = blocks_all[j * out_size:(j + 1) * out_size]
            slot_radials = ([] if fused else [r.index_select(0, slot.order) for r in radials]
                            if slot.order is not None else radials)
            xs, ws, outs = [], [], []
            cursor = 0
            for b, (m, cin, cout) in enumerate(plan.blocks):
                block = _view(packed, n, inp.prefix[m], inp.stride[m])
                if spec.radial_mode == "front" and not fused:
                    block = _scale_block(block, slot_radials[b], m > 0)
                target = _view(blocks, n, out.prefix[m], out.stride[m])
                if m == 0:
                    if b0 is not None and slot.ptr is None:
                        torch.addmm(b0, block, w0.t(), out=target)
                        continue
                    weight = w0
                else:
                    size = groups * 4 * cout * cin
                    weight = block_weights[cursor:cursor + size].view(*_pair_shape(spec, groups, cout, cin))
                    cursor += size
                xs.append(block)
                ws.append(weight)
                outs.append(target)
            _gemm_many(xs, ws, outs, slot.ptr)
            if b0 is not None and slot.ptr is not None:
                _view(blocks, n, out.prefix[0], out.stride[0]).add_(b0.index_select(0, _rows_of(slot, n)))
            if spec.radial_mode == "back":
                scaled = scaled_all[j * out_size:(j + 1) * out_size]
                for b, (m, _cin, _cout) in enumerate(plan.blocks):
                    _view(scaled, n, out.prefix[m], out.stride[m]).copy_(
                        _scale_block(_view(blocks, n, out.prefix[m], out.stride[m]), slot_radials[b], m > 0))
            slot_radials_all.append(slot_radials)
        result = _gather(scaled_all, n, out, plan, wigner, layout.rotate_out, scale=gate_scale, rows=rows,
                         copies=copies)
        tensors_to_save = [block_weights if keep_weights else None, w0 if keep_weights else None,
                           packed_all if keep_packed else None, blocks_all if keep_blocks else None,
                           x if keep_input else None, b0 if (fused and grad_gates) else None, *gates]
        if fused:
            tensors_to_save += list(radials) if (keep_input or need_x) else [None] * len(radials)
        else:
            for slot_radials in slot_radials_all:
                tensors_to_save += list(slot_radials)
        ctx.flags = (need_x, grad_w0, grad_b0, grad_pairs, grad_radials, grad_gates)
        ctx.save_for_backward(*tensors_to_save)
        ctx.state = (plan, layout, wigner, spec, n, rows, gate_scale)
        return result

    @staticmethod
    def backward(ctx, grad_out):
        ext = _ext()
        plan, layout, wigner, spec, n, rows, gate_scale = ctx.state
        need_x, grad_w0, grad_b0, grad_pairs, grad_radials, grad_gates = ctx.flags
        saved = list(ctx.saved_tensors)
        block_weights, w0, packed_all, blocks_all, x_saved, b0_saved = saved[:6]
        gates = saved[6:6 + spec.n_gates]
        radial_saved = saved[6 + spec.n_gates:]
        if spec.fused_radial:
            return _Sandwich._fused_backward(ctx, grad_out, block_weights, w0, x_saved, b0_saved, gates, radial_saved)
        inp, out = plan.inp, plan.out
        groups = max(spec.groups, 1)
        slots = spec.slots
        copies = len(slots)
        in_size, out_size = n * inp.total, n * out.total
        grad_out = grad_out.contiguous()
        front_grad = spec.radial_mode == "front" and grad_radials
        n_pair_entries = sum(groups * 4 * co * ci for m, ci, co in plan.blocks if m > 0)
        grad_block_weights = grad_out.new_empty(n_pair_entries) if grad_pairs else None
        grad_w0_value = grad_b0_value = None
        radial_grads = [None] * spec.n_radials
        gate_grads = [None] * spec.n_gates
        if gates and not grad_gates:
            # d result / d blocks carries each slot's gate as a per-edge scale of its copy.
            grad_blocks_all = _rotate(grad_out, out, plan, wigner, layout.rotate_out, scale=gate_scale, rows=rows,
                                      copies=copies)
        else:
            grad_blocks_all = _rotate(grad_out, out, plan, wigner, layout.rotate_out, rows=rows, copies=copies)
        grad_packed_all = grad_out.new_empty(copies * in_size) if need_x else None
        for j, slot in enumerate(slots):
            packed = None if packed_all is None else packed_all[j * in_size:(j + 1) * in_size]
            blocks = None if blocks_all is None else blocks_all[j * out_size:(j + 1) * out_size]
            grad_blocks = grad_blocks_all[j * out_size:(j + 1) * out_size]
            grad_packed = None if grad_packed_all is None else grad_packed_all[j * in_size:(j + 1) * in_size]
            slot_radials = radial_saved[j * spec.n_radials:(j + 1) * spec.n_radials]
            slot_rows = slot.rows
            if gates and grad_gates:
                gate = gates[j]
                row_sum = grad_out.new_zeros(n)
                for b, (m, _cin, _cout) in enumerate(plan.blocks):
                    produced = _view(blocks, n, out.prefix[m], out.stride[m])
                    if spec.radial_mode == "back":
                        produced = _scale_block(produced, slot_radials[b], m > 0)
                    row_sum += (_view(grad_blocks, n, out.prefix[m], out.stride[m]) * produced).sum(dim=1)
                gate_grads[j] = row_sum if slot_rows is None else row_sum.index_select(0, slot_rows)
                gate_rows = gate if slot.order is None else gate.index_select(0, slot.order)
                for m, _cin, _cout in plan.blocks:
                    _view(grad_blocks, n, out.prefix[m], out.stride[m]).mul_(gate_rows.unsqueeze(1))
            slot_radial_grads = [None] * spec.n_radials
            if spec.radial_mode == "back":
                for b, (m, _cin, _cout) in enumerate(plan.blocks):
                    view = _view(grad_blocks, n, out.prefix[m], out.stride[m])
                    if grad_radials:
                        slot_radial_grads[b] = _rowdot(view, _view(blocks, n, out.prefix[m], out.stride[m]), m > 0, n)
                    view.copy_(_scale_block(view, slot_radials[b], m > 0))
            dx_in, dx_w, dx_out, front_rows = [], [], [], []
            dw_grad, dw_in, dw_out = [], [], []
            cursor = 0
            for b, (m, cin, cout) in enumerate(plan.blocks):
                win = inp.stride[m]
                grad_block = _view(grad_blocks, n, out.prefix[m], out.stride[m])
                size = groups * 4 * cout * cin if m > 0 else 0
                weight = w0 if m == 0 else (block_weights[cursor:cursor + size].view(*_pair_shape(spec, groups, cout, cin))
                                            if block_weights is not None else None)
                if need_x or front_grad:
                    target = (grad_out.new_empty(n, win) if spec.radial_mode == "front"
                              else _view(grad_packed, n, inp.prefix[m], win))
                    dx_in.append(grad_block)
                    dx_w.append(weight)
                    dx_out.append(target)
                    front_rows.append((b, m, win, target))
                if (m == 0 and grad_w0) or (m > 0 and grad_pairs):
                    block = _view(packed, n, inp.prefix[m], win)
                    if spec.radial_mode == "front":
                        block = _scale_block(block, slot_radials[b], m > 0)
                    if m == 0:
                        if grad_w0_value is None:
                            grad_w0_value = (grad_out.new_empty(groups, cout, cin) if spec.groups
                                             else grad_out.new_empty(cout, cin))
                        target = grad_w0_value
                    else:
                        target = grad_block_weights[cursor:cursor + size].view(*_pair_shape(spec, groups, cout, cin))
                    dw_grad.append(grad_block)
                    dw_in.append(block)
                    dw_out.append(target)
                if m == 0 and grad_b0:
                    if spec.groups:
                        value = grad_out.new_zeros(groups, cout).index_add_(0, _rows_of(slot, n), grad_block)
                    else:
                        value = grad_block.sum(dim=0)
                    grad_b0_value = value if grad_b0_value is None else grad_b0_value + value
                cursor += size
            _gemm_many(dx_in, dx_w, dx_out, slot.ptr, transpose=True)
            if spec.radial_mode == "front":
                for b, m, win, grad_in in front_rows:
                    if front_grad:
                        slot_radial_grads[b] = _rowdot(grad_in, _view(packed, n, inp.prefix[m], win), m > 0, n)
                    if need_x:
                        _view(grad_packed, n, inp.prefix[m], win).copy_(_scale_block(grad_in, slot_radials[b], m > 0))
            _weight_grad_many(dw_grad, dw_in, dw_out, slot.ptr, accumulate=j > 0)
            for b, value in enumerate(slot_radial_grads):
                if value is None:
                    continue
                if slot_rows is not None:
                    value = value.index_select(0, slot_rows)
                radial_grads[b] = value if radial_grads[b] is None else radial_grads[b] + value
        grad_x = None
        if need_x:
            # One gather sums the input gradients of every slot and rotates them back.
            grad_x = _gather(grad_packed_all, n, inp, plan, wigner, layout.rotate_in, rows=rows, copies=copies)
        pair_grads = [None] * spec.n_pairs
        if grad_pairs:
            pair_blocks = [(ci, co) for m, ci, co in plan.blocks if m > 0]
            pair_grads = ext.block_complex_weight_grads_fp32(
                grad_block_weights, [co for _, co in pair_blocks], [ci for ci, _ in pair_blocks], spec.groups)
        grads = [grad_x, None, None, None, None]
        if spec.has_m0:
            grads.append(grad_w0_value)
        if spec.has_bias:
            grads.append(grad_b0_value)
        grads.extend(pair_grads)
        grads.extend(radial_grads)
        grads.extend(gate_grads)
        return tuple(grads)

    @staticmethod
    def _fused_backward(ctx, grad_out, block_weights, w0, x, b0, gates, radials):
        """Backward of a front-radial layer whose forward kept only the input."""
        ext = _ext()
        plan, layout, wigner, spec, n, rows, gate_scale = ctx.state
        need_x, grad_w0, grad_b0, grad_pairs, grad_radials, grad_gates = ctx.flags
        inp, out = plan.inp, plan.out
        groups = max(spec.groups, 1)
        slots = spec.slots
        copies = len(slots)
        in_size, out_size = n * inp.total, n * out.total
        grad_out = grad_out.contiguous()
        front_grad = grad_radials
        gate_grad = bool(gates) and grad_gates
        need_inputs = grad_w0 or grad_pairs or front_grad or gate_grad
        need_dx = need_x or front_grad or gate_grad
        radial_by_m = _radials_by_m(plan, radials) if radials and radials[0] is not None else ()
        # Rotated output gradient (not gate-scaled) and the recomputed radial-scaled inputs.
        grad_blocks_all = _rotate(grad_out, out, plan, wigner, layout.rotate_out, rows=rows, copies=copies)
        inputs_all = plain = None
        if need_inputs:
            rotated = _rotate(x, inp, plan, wigner, layout.rotate_in, rows=rows, copies=copies, radials=radial_by_m,
                              plain=front_grad)
            inputs_all = rotated[:copies * in_size]
            plain = rotated[copies * in_size:] if front_grad else None
        grad_packed_all = grad_out.new_empty(copies * in_size) if need_dx else None
        n_pair_entries = sum(groups * 4 * co * ci for m, ci, co in plan.blocks if m > 0)
        grad_block_weights = grad_out.new_empty(n_pair_entries) if grad_pairs else None
        grad_w0_value = grad_b0_value = None
        bias_terms = [None] * copies
        for j, slot in enumerate(slots):
            grad_blocks = grad_blocks_all[j * out_size:(j + 1) * out_size]
            inputs = None if inputs_all is None else inputs_all[j * in_size:(j + 1) * in_size]
            dx_in, dx_w, dx_out, dw_grad, dw_in, dw_out = [], [], [], [], [], []
            cursor = 0
            for m, cin, cout in plan.blocks:
                size = groups * 4 * cout * cin if m > 0 else 0
                weight = w0 if m == 0 else (block_weights[cursor:cursor + size].view(*_pair_shape(spec, groups, cout, cin))
                                            if block_weights is not None else None)
                grad_block = _view(grad_blocks, n, out.prefix[m], out.stride[m])
                if need_dx:
                    dx_in.append(grad_block)
                    dx_w.append(weight)
                    dx_out.append(_view(grad_packed_all[j * in_size:(j + 1) * in_size], n, inp.prefix[m], inp.stride[m]))
                if (m == 0 and grad_w0) or (m > 0 and grad_pairs):
                    if m == 0:
                        if grad_w0_value is None:
                            grad_w0_value = (grad_out.new_empty(groups, cout, cin) if spec.groups
                                             else grad_out.new_empty(cout, cin))
                        target = grad_w0_value
                    else:
                        target = grad_block_weights[cursor:cursor + size].view(*_pair_shape(spec, groups, cout, cin))
                    dw_grad.append(grad_block)
                    dw_in.append(_view(inputs, n, inp.prefix[m], inp.stride[m]))
                    dw_out.append(target)
                cursor += size
            # Input gradients of the slot before its gate: the gather applies the gates.
            _gemm_many(dx_in, dx_w, dx_out, slot.ptr, transpose=True)
            if gate_grad and b0 is not None:
                grad_m0 = _view(grad_blocks, n, out.prefix[0], out.stride[0])
                bias = b0 if slot.ptr is None else b0.index_select(0, _rows_of(slot, n))
                term = (grad_m0 * bias).sum(dim=1)
                bias_terms[j] = term if slot.rows is None else term.index_select(0, slot.rows)
            if gates:
                gate_rows = gates[j] if slot.order is None else gates[j].index_select(0, slot.order)
                for m, _cin, _cout in plan.blocks:
                    _view(grad_blocks, n, out.prefix[m], out.stride[m]).mul_(gate_rows.unsqueeze(1))
            _weight_grad_many(dw_grad, dw_in, dw_out, slot.ptr, accumulate=j > 0)
            if grad_b0:
                grad_m0 = _view(grad_blocks, n, out.prefix[0], out.stride[0])
                if spec.groups:
                    value = grad_out.new_zeros(groups, grad_m0.shape[1]).index_add_(0, _rows_of(slot, n), grad_m0)
                else:
                    value = grad_m0.sum(dim=0)
                grad_b0_value = value if grad_b0_value is None else grad_b0_value + value
        grad_x = None
        radial_grads = [None] * spec.n_radials
        gate_grads = [None] * spec.n_gates
        if need_dx:
            radial_grad = grad_out.new_empty(n, sum(inp.width)) if front_grad else None
            gate_dots = grad_out.new_empty(copies * n) if gate_grad else None
            # Sum of the gated slot input gradients, radial gradients and scaling, gate
            # products and the rotation back, in one pass over every copy.
            grad_x = _gather(grad_packed_all, n, inp, plan, wigner, layout.rotate_in, scale=gate_scale, rows=rows,
                             copies=copies, radials=radial_by_m, plain=plain, radial_grad=radial_grad,
                             dot_src=inputs_all if gate_grad else None, dot_out=gate_dots)
            if front_grad:
                for b, (m, _cin, _cout) in enumerate(plan.blocks):
                    offset = inp.radial_offsets[m]
                    radial_grads[b] = radial_grad[:, offset:offset + inp.width[m]]
            if gate_grad:
                for j in range(copies):
                    value = gate_dots[j * n:(j + 1) * n]
                    gate_grads[j] = value if bias_terms[j] is None else value + bias_terms[j]
        if not need_x:
            grad_x = None
        pair_grads = [None] * spec.n_pairs
        if grad_pairs:
            pair_blocks = [(ci, co) for m, ci, co in plan.blocks if m > 0]
            pair_grads = ext.block_complex_weight_grads_fp32(
                grad_block_weights, [co for _, co in pair_blocks], [ci for ci, _ in pair_blocks], spec.groups)
        grads = [grad_x, None, None, None, None]
        if spec.has_m0:
            grads.append(grad_w0_value)
        if spec.has_bias:
            grads.append(grad_b0_value)
        grads.extend(pair_grads)
        grads.extend(radial_grads)
        grads.extend(gate_grads)
        return tuple(grads)


def _slot(rows=None, order=None, ptr=None, device=None, group_rows=None):
    return SimpleNamespace(rows=rows, order=order, ptr=ptr, device=device, group_rows=group_rows)


def _apply(x, plan, layout, wigner, *, w0=None, b0=None, pair_weights=(), radials=(), slots=None, gates=(),
           groups=0):
    slots = slots or [_slot(device=x.device)]
    if gates and len(gates) != len(slots):
        raise ValueError("one gate per slot is required")
    radial_mode = ("front" if layout.front else "back") if radials else None
    fused = (radial_mode == "front" and all(r.stride(-1) == 1 for r in radials)
             and _edge_tiles_fit(plan, wigner, x.shape[1], x.device))
    spec = SimpleNamespace(has_m0=w0 is not None, has_bias=b0 is not None, n_pairs=len(pair_weights),
                           radial_mode=radial_mode, n_radials=len(radials), n_gates=len(gates),
                           groups=int(groups), slots=slots, fused_radial=fused)
    tensors = ([w0] if w0 is not None else []) + ([b0] if b0 is not None else []) + list(pair_weights)
    tensors += list(radials) + list(gates)
    return _Sandwich.apply(x.contiguous(), plan, layout, wigner, spec, *tensors)


def block_sandwich(x, layout, wigner, weights_by_m, radial_parts=None):
    """m>0 contribution [N, out_dim] of one [2*Cout_m, Cin_m] pair weight per m.

    ``weights_by_m`` and ``radial_parts`` are indexed by m (index 0 unused); radial
    weights are [N, Cin_m] for front layouts and [N, Cout_m] otherwise. Callers
    check ``sandwich_plan(...).supported`` first."""
    plan = sandwich_plan(layout, x.shape[1], x.device)
    if not plan.blocks:
        return x.new_zeros((x.shape[0], layout.out_dim))
    weights = tuple(weights_by_m[m] for m, _, _ in plan.blocks)
    radials = () if radial_parts is None else tuple(radial_parts[m] for m, _, _ in plan.blocks)
    return _apply(x, plan, layout, wigner, pair_weights=weights, radials=radials)


def grouped_sandwich(x, layout, wigner, weights_by_m, radial_parts, ptr_edges, order, rows):
    """m>0 contribution of [G, 2*Cout_m, Cin_m] pair weights: edges sorted by group
    (``order``: sorted position -> edge, ``rows``: edge -> sorted position, both None
    when the edges are already sorted) and group g owning the sorted rows
    ``ptr_edges[g]:ptr_edges[g+1]`` (CPU int64)."""
    plan = sandwich_plan(layout, x.shape[1], x.device)
    if not plan.blocks:
        return x.new_zeros((x.shape[0], layout.out_dim))
    weights = tuple(weights_by_m[m] for m, _, _ in plan.blocks)
    radials = () if radial_parts is None else tuple(radial_parts[m] for m, _, _ in plan.blocks)
    return _apply(x, plan, layout, wigner, pair_weights=weights, radials=radials,
                  slots=[_slot(rows, order, ptr_edges, x.device)], groups=int(weights[0].shape[0]))


def full_sandwich(x, layout, wigner, w0, b0, pair_weights_by_m, radials_by_m=None, gates=(), slots=None):
    """Whole layer output [N, out_dim], m = 0 included, summed over routing slots.

    Ungrouped: ``w0`` [Cout_0, Cin_0], ``b0`` [Cout_0], ``pair_weights_by_m[m]``
    [2*Cout_m, Cin_m]. With ``slots`` (top-k routing, one per slot, rows sorted by
    expert) the weights carry a leading expert dimension. ``radials_by_m[m]`` are
    optional radial weights of each block (front or back by ``layout.front``),
    ``gates`` optional per-edge [N] scales, one per slot. Returns None when the
    layout has no m = 0 block."""
    plan = sandwich_plan(layout, x.shape[1], x.device, with_m0=True)
    if not plan.blocks or plan.blocks[0][0] != 0:
        return None
    pair_weights = tuple(pair_weights_by_m[m] for m, _, _ in plan.blocks if m > 0)
    radials = () if radials_by_m is None else tuple(radials_by_m[m] for m, _, _ in plan.blocks)
    groups = int(w0.shape[0]) if slots is not None else 0
    return _apply(x, plan, layout, wigner, w0=w0, b0=b0, pair_weights=pair_weights, radials=radials,
                  gates=tuple(gates), slots=slots, groups=groups)
