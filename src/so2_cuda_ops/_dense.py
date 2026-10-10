"""True-dense SO(2) pair arithmetic behind the tensor-only integration API."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F


def _plan(layout, device):
    cached = getattr(layout, "_true_dense_plan", None)
    if cached is not None:
        return cached
    from .tensor_product import _multi_output_entry_map

    values, in_bases, in_ls, out_bases, out_ls = [], [], [], [], []
    cin_prefix, cout_prefix = [0], [0]
    for m, (in_base, in_l, out_base, out_l, _) in enumerate(layout.maps[1:], 1):
        cin, cout = in_base.numel(), out_base.numel()
        if not cin or not cout:
            continue
        values.append(m)
        in_bases.append(in_base)
        in_ls.append(in_l)
        out_bases.append(out_base)
        out_ls.append(out_l)
        cin_prefix.append(cin_prefix[-1] + cin)
        cout_prefix.append(cout_prefix[-1] + cout)
    with torch.inference_mode(False), torch.no_grad():
        cached = SimpleNamespace(
            values=tuple(values), in_bases=in_bases, in_ls=in_ls,
            out_bases=out_bases, out_ls=out_ls, cin_prefix=cin_prefix,
            cin_prefix_t=torch.tensor(cin_prefix, dtype=torch.long, device=device),
            cout_prefix_t=torch.tensor(cout_prefix, dtype=torch.long, device=device),
            m_values_t=torch.tensor(values, dtype=torch.long, device=device),
            entries=_multi_output_entry_map(layout, values, out_bases, out_ls, layout.out_dim, device),
        )
    layout._true_dense_plan = cached
    return cached


def _input_channel_plan(layout, plan, in_dim, device):
    """Channel tables of the pack's channel-major backward for an input width."""
    cache = plan.__dict__.setdefault("input_channel_plans", {})
    hit = cache.get(int(in_dim))
    if hit is None:
        from .tensor_product import channel_plan
        all_bases, all_ls = layout.maps[0][0], layout.maps[0][1]
        hit = channel_plan(all_bases, all_ls, plan.in_bases, plan.cin_prefix[:-1], plan.values, in_dim, device)
        cache[int(in_dim)] = hit
    return hit


class _PairSandwich(torch.autograd.Function):
    """Multi-m pack, one GEMM per m block on its view of the packed rows, and the
    output-major scatter, with the backward written out.

    Every m block stays a column range of one packed [N, 2, sum Cin] buffer: each
    GEMM reads its block with the buffer's row stride, and the backward writes each
    block's input gradient into its columns of one gradient buffer that the pack's
    channel-major gather consumes. No block is copied and no full-width gradient is
    formed per block. Front radial weights scale a block before its GEMM; the scaled
    block is recomputed for the weight gradient instead of being kept."""

    @staticmethod
    def forward(ctx, x, plan, layout, wigner, in_plan, n_blocks, *tensors):
        from . import tensor_product as tp
        weights = tensors[:n_blocks]
        radials = tensors[n_blocks:]
        offsets = layout.maps[plan.values[0]][-1]
        packed = tp._pack_pairs_multi_cuda(
            x, wigner.values, plan.in_bases, plan.in_ls, offsets, wigner.compact_offsets,
            plan.cin_prefix_t, plan.m_values_t, layout.rotate_in, wigner.mode, wigner.stride)
        n = x.shape[0]
        rows = packed.view(2 * n, packed.shape[2])
        raws = []
        for i, weight in enumerate(weights):
            a, b = plan.cin_prefix[i], plan.cin_prefix[i + 1]
            block = rows[:, a:b]
            if radials:
                block = (packed[:, :, a:b] * radials[i].unsqueeze(1)).view(2 * n, b - a)
            raws.append(torch.mm(block, weight.t()).view(n, 2, weight.shape[0]))
        out = tp._scatter_raw_pairs_multi_output_major_forward_cuda(
            raws, wigner.values, offsets, wigner.compact_offsets, plan.cout_prefix_t,
            plan.m_values_t, *plan.entries, layout.out_dim, layout.rotate_out,
            wigner.mode, wigner.stride)
        keep_packed = any(ctx.needs_input_grad[6:])
        ctx.save_for_backward(packed if keep_packed else None, *tensors)
        ctx.state = (plan, layout, wigner, in_plan, n_blocks, offsets, int(x.shape[1]), packed.shape)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        from . import tensor_product as tp
        plan, layout, wigner, in_plan, n_blocks, offsets, in_dim, packed_shape = ctx.state
        saved = ctx.saved_tensors
        packed, weights, radials = saved[0], saved[1:1 + n_blocks], saved[1 + n_blocks:]
        needs = ctx.needs_input_grad
        n = packed_shape[0]
        grad_raws = tp._raw_pairs_multi_output_grad_cuda(
            grad_out.contiguous(), wigner.values, list(plan.out_bases), list(plan.out_ls), offsets,
            wigner.compact_offsets, plan.cout_prefix_t, plan.m_values_t, layout.rotate_out,
            wigner.mode, wigner.stride)
        grad_packed = grad_out.new_empty(packed_shape) if needs[0] else None
        grad_rows = None if grad_packed is None else grad_packed.view(2 * n, packed_shape[2])
        rows = None if packed is None else packed.view(2 * n, packed_shape[2])
        grad_weights = [None] * n_blocks
        grad_radials = [None] * len(radials)
        for i, weight in enumerate(weights):
            a, b = plan.cin_prefix[i], plan.cin_prefix[i + 1]
            grad_raw = grad_raws[i].view(2 * n, weight.shape[0])
            radial_grad = bool(radials) and needs[6 + n_blocks + i]
            if radials and (needs[0] or radial_grad):
                grad_block = torch.mm(grad_raw, weight).view(n, 2, b - a)
                if radial_grad:
                    grad_radials[i] = (grad_block * packed[:, :, a:b]).sum(dim=1)
                if needs[0]:
                    torch.mul(grad_block, radials[i].unsqueeze(1), out=grad_packed[:, :, a:b])
            elif needs[0]:
                torch.mm(grad_raw, weight, out=grad_rows[:, a:b])
            if needs[6 + i]:
                block = rows[:, a:b]
                if radials:
                    block = (packed[:, :, a:b] * radials[i].unsqueeze(1)).view(2 * n, b - a)
                grad_weights[i] = torch.mm(grad_raw.t(), block)
        grad_x = None
        if needs[0]:
            grad_x = tp._channel_pack_grad_cuda(
                grad_packed, wigner.values, offsets, wigner.compact_offsets, in_plan, in_dim,
                layout.rotate_in, wigner.mode, wigner.stride)
        return (grad_x, None, None, None, None, None, *grad_weights, *grad_radials)


def pair_sandwich(x, layout, wigner, weights_by_m, radial_parts=None):
    """m>0 contribution of one weight per m block through ``_PairSandwich``.

    ``weights_by_m[m]`` is the [2*Cout, Cin] pair weight of block m (index 0 unused);
    ``radial_parts[m]`` scales the block's input channels (front layouts only)."""
    plan = _plan(layout, x.device)
    if not plan.values:
        return None
    in_plan = _input_channel_plan(layout, plan, x.shape[1], x.device)
    weights = tuple(weights_by_m[m] for m in plan.values)
    radials = () if radial_parts is None else tuple(radial_parts[m] for m in plan.values)
    return _PairSandwich.apply(x.contiguous(), plan, layout, wigner, in_plan, len(weights), *weights, *radials)


def true_dense_pairs(x, layout, wigner, linears, radial_parts=None):
    """Return ordered m>0 contributions using the qualified raw epilogue.

    ``linears`` includes m=0 and contains non-routed ``LinearWeights`` with
    [2*Cout, Cin] pair weights. The caller computes m=0 first and adds returned
    tensors in order. With radial weights after the linear layer, the original
    per-m pack/scatter arithmetic is retained. Unsupported inputs return None
    before any native execution; native errors propagate to the caller.
    """
    from .deeptb import _supported
    if not _supported(x, wigner):
        return None
    if len(linears) != len(layout.maps) or (radial_parts is not None and len(radial_parts) < len(linears)):
        return None
    for m, params in enumerate(linears[1:], 1):
        cin, _, cout, _, _ = layout.maps[m]
        weight = params.weight
        if (params.routed or params.bias is not None or weight.ndim != 2
                or weight.shape != (2 * cout.numel(), cin.numel())
                or weight.device != x.device or weight.dtype != x.dtype):
            return None
        if radial_parts is not None:
            radial = radial_parts[m]
            width = cin.numel() if layout.front else cout.numel()
            if (radial.shape != (x.shape[0], width) or radial.device != x.device
                    or radial.dtype != x.dtype):
                return None
    from .tensor_product import _PackPairFunction, _ScatterPairOutputFunction
    if radial_parts is not None and not layout.front:
        outputs = []
        for m, params in enumerate(linears[1:], 1):
            in_base, in_l, out_base, out_l, offsets = layout.maps[m]
            pair = _PackPairFunction.apply(
                x.contiguous(), wigner.values, in_base, in_l, offsets,
                wigner.compact_offsets, m, layout.rotate_in, wigner.mode, wigner.stride,
            )
            raw = F.linear(pair, params.weight)
            width = params.weight.shape[0] // 2
            row0, row1 = raw.unflatten(-1, (2, width)).unbind(1)
            r0, i0 = row0.unbind(-2)
            r1, i1 = row1.unbind(-2)
            result = torch.stack((r0 - i1, r1 + i0), dim=1)
            outputs.append(_ScatterPairOutputFunction.apply(
                (result * radial_parts[m].unsqueeze(1)).contiguous(), wigner.values,
                out_base, out_l, offsets, wigner.compact_offsets, layout.out_dim,
                m, layout.rotate_out, wigner.mode, wigner.stride,
            ))
        return tuple(outputs)
    contribution = pair_sandwich(x, layout, wigner, tuple(params.weight for params in linears), radial_parts)
    return () if contribution is None else (contribution,)
