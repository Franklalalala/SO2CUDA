"""True-dense SO(2) pair arithmetic behind the tensor-only integration API."""


def _fits(tensor, shape, x):
    return tensor.shape == shape and tensor.device == x.device and tensor.dtype == x.dtype


def true_dense_pairs(x, layout, wigner, linears, radial_parts=None, *, include_m0=False):
    """Return ordered m>0 contributions using the qualified raw epilogue.

    ``linears`` includes m=0 and contains non-routed ``LinearWeights`` with
    [2*Cout, Cin] pair weights. The caller computes m=0 first and adds returned
    tensors in order. Radial weights scale each block's input channels for front
    layouts and its output channels otherwise. Unsupported inputs return None
    before any native execution; native errors propagate to the caller.

    With ``include_m0=True`` the m=0 term (the [Cout_0, Cin_0] weight of
    ``linears[0]``, its optional bias and ``radial_parts[0]``) is computed with the
    pairs, and the single returned tensor is the whole layer output.
    """
    from .deeptb import _supported
    if not _supported(x, wigner):
        return None
    if len(linears) != len(layout.maps) or (radial_parts is not None and len(radial_parts) < len(linears)):
        return None
    first = 0 if include_m0 else 1
    for m, params in enumerate(linears[first:], first):
        cin, _, cout, _, _ = layout.maps[m]
        weight = params.weight
        if params.routed or weight.ndim != 2 or not _fits(weight, ((2 if m else 1) * cout.numel(), cin.numel()), x):
            return None
        if params.bias is not None and (m > 0 or not _fits(params.bias, (cout.numel(),), x)):
            return None
        if radial_parts is not None:
            width = cin.numel() if layout.front else cout.numel()
            if not _fits(radial_parts[m], (x.shape[0], width), x):
                return None
    from ._sandwich import block_sandwich, full_sandwich, sandwich_plan
    if not sandwich_plan(layout, x.shape[1], x.device, with_m0=include_m0).supported:
        return None
    if include_m0:
        out = full_sandwich(x, layout, wigner, linears[0].weight, linears[0].bias,
                            tuple(params.weight for params in linears), radial_parts)
        return None if out is None else (out,)
    return (block_sandwich(x, layout, wigner, tuple(params.weight for params in linears), radial_parts),)
