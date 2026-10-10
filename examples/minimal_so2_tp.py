#!/usr/bin/env python3
"""Apply a complete edge SO(2) tensor product through the public tensor-only API.

The helpers in this example prepare e3nn mul_ir features and constant geometry;
they are example code, not additional package API. Reuse layout and Wigner data
when the layer and edge directions stay unchanged.
"""
from __future__ import annotations

import argparse
import math

import torch
from e3nn import o3
from so2_cuda_ops.deeptb import LinearWeights, prepare_layout, prepare_wigner, true_dense_pairs


def prepare_so2_tp(irreps_in, irreps_out, edge_vectors, m_max=None):
    """Return reusable public layout and Wigner descriptors for the supplied edges."""
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    l_max = max(irreps_in.lmax, irreps_out.lmax)
    m_max = min(irreps_in.lmax, irreps_out.lmax) if m_max is None else m_max
    if not 0 <= m_max <= min(irreps_in.lmax, irreps_out.lmax):
        raise ValueError("m_max must lie in the angular range of both irreps")
    if edge_vectors.ndim != 2 or edge_vectors.shape[1] != 3:
        raise ValueError("edge_vectors must have shape [edges, 3]")
    if edge_vectors.requires_grad:
        raise ValueError("This CUDA interface requires constant geometry")
    if not bool(torch.isfinite(edge_vectors).all()) or bool((edge_vectors.norm(dim=1) == 0).any()):
        raise ValueError("Edge vectors must be finite and nonzero")
    entries = lambda irreps: tuple((ir.l, mul, sl.start) for (mul, ir), sl in zip(irreps, irreps.slices()))
    layout = prepare_layout(entries(irreps_in), entries(irreps_out), m_max=m_max,
                            l_max=l_max, out_dim=irreps_out.dim, device=edge_vectors.device)
    # Prepare geometry in FP64, then round once to the feature dtype. e3nn creates
    # Wigner generators using the default dtype, so restore it after preparation.
    alpha, beta = o3.xyz_to_angles(edge_vectors.double())
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(edge_vectors.device):
            blocks = tuple(o3.wigner_D(l, alpha, beta, torch.zeros_like(alpha))
                           .to(edge_vectors.dtype).contiguous() for l in range(l_max + 1))
    finally:
        torch.set_default_dtype(previous_dtype)
    reference = edge_vectors.new_empty((len(edge_vectors), irreps_in.dim))
    wigner = prepare_wigner(reference, blocks, l_max=l_max)
    return layout, wigner


def make_weights(irreps_in, irreps_out, m_max, *, device, dtype=torch.float32):
    """Create trainable W0 and [A_m; B_m] weights, shared by every edge."""
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    weights = []
    for m in range(m_max + 1):
        cin = sum(mul for mul, ir in irreps_in if ir.l >= m)
        cout = sum(mul for mul, ir in irreps_out if ir.l >= m)
        if min(cin, cout) <= 0:
            raise ValueError("Each retained m block needs input and output channels")
        width = cout * (2 if m else 1)
        weight = torch.nn.Parameter(torch.randn(width, cin, device=device, dtype=dtype)
                                    / math.sqrt(cin * (2 if m else 1)))
        weights.append(LinearWeights(weight, routed=False))
    return tuple(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irreps-in", default="32x0e+32x1o+32x2e")
    parser.add_argument("--irreps-out", help="Defaults to the input irreps")
    parser.add_argument("--mmax", type=int, help="Defaults to the full angular range")
    parser.add_argument("--edges", type=int, default=512)
    args = parser.parse_args()
    if args.edges <= 0:
        parser.error("--edges must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    irreps_in = o3.Irreps(args.irreps_in)
    irreps_out = o3.Irreps(args.irreps_out or args.irreps_in)
    m_max = min(irreps_in.lmax, irreps_out.lmax) if args.mmax is None else args.mmax
    vectors = torch.randn(args.edges, 3, device="cuda", dtype=torch.float32)
    x = torch.randn(args.edges, irreps_in.dim, device="cuda", dtype=torch.float32, requires_grad=True)
    layout, wigner = prepare_so2_tp(irreps_in, irreps_out, vectors, m_max)
    weights = make_weights(irreps_in, irreps_out, m_max, device=x.device, dtype=x.dtype)
    parts = None if wigner is None else true_dense_pairs(x, layout, wigner, weights, include_m0=True)
    if parts is None:
        raise RuntimeError("The CUDA interface declined this input; the caller must use its reference implementation")
    y = parts[0]
    y.square().mean().backward()
    print({"input_shape": tuple(x.shape), "output_shape": tuple(y.shape),
           "input_gradient_finite": bool(torch.isfinite(x.grad).all()),
           "weight_gradients_finite": all(w.weight.grad is not None and bool(torch.isfinite(w.weight.grad).all())
                                           for w in weights)})


if __name__ == "__main__":
    main()
