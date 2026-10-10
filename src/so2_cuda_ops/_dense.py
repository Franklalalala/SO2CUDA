"""True-dense SO(2) pair arithmetic behind the tensor-only integration API."""


def true_dense_pairs(x, layout, wigner, linears, radial_parts=None):
    """Return ordered m>0 contributions using the qualified raw epilogue.

    ``linears`` includes m=0 and contains non-routed ``LinearWeights`` with
    [2*Cout, Cin] pair weights. The caller computes m=0 first and adds returned
    tensors in order. Radial weights scale each block's input channels for front
    layouts and its output channels otherwise. Unsupported inputs return None
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
    from ._sandwich import block_sandwich, sandwich_plan
    if not sandwich_plan(layout, x.shape[1], x.device).supported:
        return None
    return (block_sandwich(x, layout, wigner, tuple(params.weight for params in linears), radial_parts),)
