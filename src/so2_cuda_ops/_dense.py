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
    from .tensor_product import (
        _PackPairFunction, _PackPairsMultiFunction,
        _ScatterPairOutputFunction, _ScatterRawPairsMultiOutputMajorFunction,
    )
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
    plan = _plan(layout, x.device)
    if not plan.values:
        return ()
    offsets = layout.maps[plan.values[0]][-1]
    packed = _PackPairsMultiFunction.apply(
        x.contiguous(), wigner.values, plan.in_bases, plan.in_ls, offsets,
        wigner.compact_offsets, plan.cin_prefix_t, plan.m_values_t,
        layout.rotate_in, wigner.mode, wigner.stride,
    )
    raw = []
    for i, m in enumerate(plan.values):
        pair = packed[:, :, plan.cin_prefix[i]:plan.cin_prefix[i + 1]]
        if radial_parts is not None:
            pair = pair * radial_parts[m].unsqueeze(1)
        raw.append(F.linear(pair, linears[m].weight).contiguous())
    contribution = _ScatterRawPairsMultiOutputMajorFunction.apply(
        wigner.values, offsets, wigner.compact_offsets,
        plan.cout_prefix_t, plan.m_values_t, *plan.entries, layout.out_dim,
        layout.rotate_out, wigner.mode, wigner.stride, len(raw),
        *raw, *plan.out_bases, *plan.out_ls,
    )
    return (contribution,)
