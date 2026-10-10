"""Comparable edge SO(2) convolutions using shared, canonical parameters.

Geometry is prepared outside forward. Features use e3nn mul_ir layout;
canonical pair weights contain [A; B] and the pair order is (-m,+m).
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
from operator_layout import FeatureLayout, NativeOperator, NativeWigner


def uniform_irreps(lmax, channels):
    return o3.Irreps([(channels, (l, (-1) ** l)) for l in range(lmax + 1)])


def canonical_weights(irreps_in, irreps_out, mmax, *, device, dtype=torch.float32):
    result = []
    for m in range(mmax + 1):
        ni = sum(mul for mul, ir in o3.Irreps(irreps_in) if ir.l >= m)
        no = sum(mul for mul, ir in o3.Irreps(irreps_out) if ir.l >= m)
        result.append(torch.randn(no * (2 if m else 1), ni, device=device, dtype=dtype)
                      / math.sqrt(ni * (2 if m else 1)))
    return tuple(result)


@dataclass
class Geometry:
    vectors: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    rotation_matrix: torch.Tensor
    blocks: tuple


def prepare_geometry(vectors, lmax):
    # Evaluate geometry in float64 then round, avoiding large-l matrix-exp error.
    dtype = vectors.dtype
    a, b = o3.xyz_to_angles(vectors.double())
    z = torch.zeros_like(a)
    # e3nn 0.5.8 creates its generators on the default device/dtype. Scope both
    # defaults to this preparation call and restore dtype before returning.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(vectors.device):
            blocks = tuple(o3.wigner_D(l, a, b, z).to(dtype).contiguous() for l in range(lmax + 1))
    finally:
        torch.set_default_dtype(previous_dtype)
    return Geometry(vectors, a.to(dtype), b.to(dtype), o3.angles_to_matrix(a, b, z).to(dtype), blocks)


def rotate_features(x, irreps, blocks, inverse=False):
    parts = []
    for (mul, ir), part in zip(irreps, x.split([mul * ir.dim for mul, ir in irreps], dim=-1)):
        matrix = blocks[ir.l].transpose(-1, -2) if inverse else blocks[ir.l]
        parts.append(torch.bmm(part.reshape(len(x), mul, ir.dim), matrix).flatten(1))
    return torch.cat(parts, dim=-1)


class CanonicalOperator(nn.Module):
    def __init__(self, irreps_in, irreps_out, mmax, weights, geometry):
        super().__init__()
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        self.mmax, self.geometry = mmax, geometry
        self.weights = nn.ParameterList(nn.Parameter(w.detach().clone()) for w in weights)
        self.metadata = {}

    def canonical_gradients(self):
        return tuple(w.grad for w in self.weights)

    def forward_native(self, x):
        return self.forward(x)

    def input_to_native(self, x):
        return x

    output_to_native = input_to_native
    input_from_native = input_to_native
    output_from_native = input_to_native


class NaiveOperator(CanonicalOperator):
    """Our implementation of upstream SO2_Linear, with geometry precomputed."""

    def __init__(self, *args):
        super().__init__(*args)
        for name, irreps in (("in", self.irreps_in), ("out", self.irreps_out)):
            masks = torch.zeros(self.mmax + 1, irreps.dim, dtype=torch.bool, device=self.weights[0].device)
            for (mul, ir), sl in zip(irreps, irreps.slices()):
                starts = sl.start + torch.arange(mul, device=masks.device) * ir.dim
                for m in range(min(ir.l, self.mmax) + 1):
                    masks[m, starts + ir.l - m] = True
                    masks[m, starts + ir.l + m] = True
            self.register_buffer(name + "_mask", masks, persistent=False)
        self.geometry = Geometry(None, None, None, None, self.geometry.blocks)
        self.metadata = {"feature_layout": "e3nn mul_ir", "geometry": "compact per-l Wigner blocks"}

    def forward(self, x):
        n = len(x)
        local = torch.zeros_like(x)
        groups = defaultdict(list)
        for (mul, ir), sl in zip(self.irreps_in, self.irreps_in.slices()):
            groups[ir.l].append((mul, sl))
            if ir.l == 0:
                local[:, sl] = x[:, sl]
        for l, group in groups.items():
            if l == 0:
                continue
            parts = [x[:, sl].reshape(n, mul, 2*l+1) for mul, sl in group]
            transformed = torch.bmm(torch.cat(parts, dim=1), self.geometry.blocks[l])
            for part, (_, sl) in zip(transformed.split([mul for mul, _ in group], dim=1), group):
                local[:, sl] = part.flatten(1)
        out = x.new_zeros(n, self.irreps_out.dim)
        for m, weight in enumerate(self.weights):
            inp = local[:, self.in_mask[m]]
            if m:
                inp = inp.reshape(n, -1, 2).transpose(1, 2).contiguous()
            raw = F.linear(inp, weight)
            if m:
                width = weight.shape[0] // 2
                real, imag = raw.narrow(2, 0, width), raw.narrow(2, width, width)
                value = torch.cat((real.narrow(1, 0, 1) - imag.narrow(1, 1, 1),
                                   real.narrow(1, 1, 1) + imag.narrow(1, 0, 1)), dim=1)
                raw = value.transpose(1, 2).contiguous().reshape(n, -1)
            out[:, self.out_mask[m]] += raw
        for (mul, ir), sl in zip(self.irreps_out, self.irreps_out.slices()):
            if ir.l:
                part = out[:, sl].reshape(n, mul, ir.dim)
                out[:, sl] = torch.einsum("nij,nmj->nmi", self.geometry.blocks[ir.l], part).reshape(n, -1)
        return out.contiguous()


class SO2CUDAOperator(CanonicalOperator):
    """Public pair APIs, the m=0 term computed in the same call (``include_m0=True``)."""

    def __init__(self, *args, candidate="dense_pairs"):
        super().__init__(*args)
        if candidate not in SO2CUDA_CANDIDATES:
            raise ValueError(f"Unknown SO2CUDA candidate: {candidate}")
        self.candidate = candidate
        from so2_cuda_ops.deeptb import prepare_layout, prepare_wigner, DenseRouting
        device = self.weights[0].device
        entries = lambda irreps: tuple((ir.l, mul, sl.start) for (mul, ir), sl in zip(irreps, irreps.slices()))
        lmax = max(self.irreps_in.lmax, self.irreps_out.lmax)
        self.layout = prepare_layout(entries(self.irreps_in), entries(self.irreps_out),
                                     m_max=self.mmax, l_max=lmax, out_dim=self.irreps_out.dim, device=device)
        n = len(self.geometry.vectors)
        edge_reference = self.geometry.vectors.new_empty((n, self.irreps_in.dim))
        self.wigner = prepare_wigner(edge_reference, self.geometry.blocks, l_max=lmax)
        self.routing = (DenseRouting(torch.zeros(n, device=device, dtype=torch.long),
                                     torch.tensor([0, n*2], dtype=torch.long))
                        if candidate != "true_dense_pairs" else None)
        if self.wigner is None:
            raise ValueError("SO2CUDA requires CUDA FP32 and constant geometry")
        self.metadata = {"candidate": candidate,
                         "api": "so2_cuda_ops.deeptb." + (
                             "true_dense_pairs" if candidate == "true_dense_pairs" else "dense_pairs"),
                         "forward_mode": ("indexed_sandwich_multi_grouped" if candidate == "dense_pairs_grouped"
                                          else "default" if candidate == "dense_pairs" else None)}
        self.metadata["feature_layout"] = "e3nn mul_ir"
        self.metadata["m0"] = "include_m0=True: m=0 runs inside the public pair call"
        # m=0 no longer needs the per-l blocks; only the packed Wigner data stay alive.
        self.geometry = Geometry(None, None, None, None, ())

    def forward(self, x):
        from so2_cuda_ops.deeptb import dense_pairs, true_dense_pairs, LinearWeights
        if self.candidate == "true_dense_pairs":
            parts = true_dense_pairs(x, self.layout, self.wigner,
                                     tuple(LinearWeights(w, routed=False) for w in self.weights), None,
                                     include_m0=True)
        else:
            parts = dense_pairs(x, self.layout, self.wigner, tuple(w.unsqueeze(0) for w in self.weights), None,
                                self.routing, include_m0=True,
                                forward_mode=("indexed_sandwich_multi_grouped"
                                              if self.candidate == "dense_pairs_grouped" else None))
        if parts is None:
            raise RuntimeError("SO2CUDA public API declined this configuration")
        return parts[0]


SO2CUDA_CANDIDATES = ("dense_pairs", "dense_pairs_grouped", "true_dense_pairs")


class ActivationOperator(CanonicalOperator):
    """Public activation_forward (the PDQ-MoE entry) with one expert, top-1 and gate 1.

    Every m block, m=0 included, is a routed LinearWeights holding the canonical weight
    as a [1, out, in] view; the routing is the slot layout of an all-zero expert index.
    Routing metadata is built once, outside the timed call."""

    def __init__(self, *args):
        super().__init__(*args)
        from so2_cuda_ops.deeptb import ActivationRouting, prepare_layout, prepare_wigner
        device = self.weights[0].device
        entries = lambda irreps: tuple((ir.l, mul, sl.start) for (mul, ir), sl in zip(irreps, irreps.slices()))
        lmax = max(self.irreps_in.lmax, self.irreps_out.lmax)
        self.layout = prepare_layout(entries(self.irreps_in), entries(self.irreps_out),
                                     m_max=self.mmax, l_max=lmax, out_dim=self.irreps_out.dim, device=device)
        n = len(self.geometry.vectors)
        self.wigner = prepare_wigner(self.geometry.vectors.new_empty((n, self.irreps_in.dim)),
                                     self.geometry.blocks, l_max=lmax)
        if self.wigner is None:
            raise ValueError("SO2CUDA requires CUDA FP32 and constant geometry")
        index = torch.zeros(n, 1, dtype=torch.long, device=device)
        order = torch.arange(n, device=device)
        slot = (order, order, torch.tensor([0, n], dtype=torch.long), index[:, 0])
        self.routing = ActivationRouting(index, torch.ones(n, 1, device=device), (slot,))
        self.metadata = {"candidate": "activation_forward", "api": "so2_cuda_ops.deeptb.activation_forward",
                         "feature_layout": "e3nn mul_ir",
                         "routing": "one expert, top-1, gate 1; every m block routed, m=0 included"}
        # m=0 runs inside the call, so the per-l blocks are not kept.
        self.geometry = Geometry(None, None, None, None, ())

    def forward(self, x):
        from so2_cuda_ops.deeptb import LinearWeights, activation_forward
        out = activation_forward(x, self.layout, self.wigner, tuple(LinearWeights(w.unsqueeze(0)) for w in self.weights),
                                 None, self.routing)
        if out is None:
            raise RuntimeError("SO2CUDA activation_forward declined this configuration")
        return out


@contextmanager
def so2cuda_candidate_environment(candidate):
    """Select the public environment control outside the measured region.

    The current dense_pairs keyword selects the multi-m family, but the family
    implementation reads the environment again to select grouped packing.
    Restore both the public control and its synchronized legacy alias.
    """
    names = ("SO2_CUDA_FORWARD_MODE", "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE")
    previous = {name: os.environ.get(name) for name in names}
    mode = ("indexed_sandwich_multi_grouped" if candidate == "dense_pairs_grouped"
            else "indexed_sandwich_multi")
    try:
        for name in names:
            os.environ[name] = mode
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ExplicitGEMMOperator(NativeOperator, CanonicalOperator):
    """Full coefficient Wigner bmm and public grouped GEMM, without fused packing.

    This ablation uses the same positive-m grouped GEMM as dense_pairs. The
    coefficient layout is materialized before and after the linear operation;
    all layout changes use PyTorch indexing, never the indexed sandwich kernels.
    """

    def __init__(self, *args):
        super().__init__(*args)
        from operator_eqv3 import _uniform_layout, UnsupportedConfiguration
        _, self.cin, lin = _uniform_layout(self.irreps_in, "input")
        _, self.cout, lout = _uniform_layout(self.irreps_out, "output")
        if lin != lout:
            raise UnsupportedConfiguration("Explicit GEMM ablation requires equal input/output lmax")
        self.lmax = lin
        n, width = len(self.geometry.vectors), (lin + 1) ** 2
        matrix = self.weights[0].new_zeros(n, width, width)
        for l, block in enumerate(self.geometry.blocks):
            matrix[:, l*l:(l+1)**2, l*l:(l+1)**2] = block.transpose(1, 2)
        self.register_buffer("ptr", torch.tensor([0, 2*n], dtype=torch.long), persistent=False)
        order = []
        self.m_widths = []
        for m in range(self.mmax + 1):
            indices = [l*l+l+sign*m for sign in ((0,) if m == 0 else (-1, 1))
                       for l in range(m, lin+1)]
            order.extend(indices)
            self.m_widths.append(len(indices))
        # Merge the m permutation into the full Wigner matrix, just as the
        # original explicit-rotation baseline does during geometry setup.
        self.dropped_width = width - len(order)
        order.extend(i for i in range(width) if i not in order)
        self.register_buffer("wigner_full", matrix.index_select(
            1, torch.tensor(order, device=matrix.device)), persistent=False)
        self.input_layout = FeatureLayout(self.irreps_in, matrix.device,
            groups=[[i for i, _ in sorted(enumerate(self.irreps_in), key=lambda item: item[1].ir.l)]],
            coefficient_shape=True)
        self.output_layout = FeatureLayout(self.irreps_out, matrix.device,
            groups=[[i for i, _ in sorted(enumerate(self.irreps_out), key=lambda item: item[1].ir.l)]],
            coefficient_shape=True)
        self.geometry = None
        self.metadata = {
            "feature_layout": "native [edges, (lmax+1)^2, channels]; conversions outside timing",
            "api": "torch.bmm + so2_cuda_ops.grouped_gemm_multi + torch.bmm",
            "geometry": "full coefficient Wigner matrix, precomputed",
            "gemm": "same positive-m grouped_gemm_multi as dense_pairs; m0 uses F.linear",
            "packing": "precomputed Wigner m permutation and split; no indexed sandwich pack/scatter",
        }

    def forward_native(self, x):
        from so2_cuda_ops import grouped_gemm_multi
        n = len(x)
        local = torch.bmm(self.wigner_full, x)
        splits = self.m_widths + ([self.dropped_width] if self.dropped_width else [])
        by_m = local.split(splits, dim=1)
        m0 = F.linear(by_m[0].flatten(1), self.weights[0])
        pair_inputs = [part.reshape(n*2, -1).contiguous() for part in by_m[1:self.mmax+1]]
        raw = grouped_gemm_multi(pair_inputs, [self.ptr]*self.mmax,
                                 [w.unsqueeze(0) for w in self.weights[1:]]) if pair_inputs else []
        parts = [m0.reshape(n, self.lmax+1, self.cout)]
        for m, value in enumerate(raw, 1):
            width = self.weights[m].shape[0] // 2
            value = value.reshape(n, 2, 2*width)
            real, imag = value.split(width, dim=-1)
            pair = torch.stack((real[:, 0] - imag[:, 1], real[:, 1] + imag[:, 0]), dim=1)
            parts.append(pair.reshape(n, -1, self.cout))
        if self.dropped_width:
            parts.append(x.new_zeros(n, self.dropped_width, self.cout))
        ordered = torch.cat(parts, dim=1)
        return torch.bmm(self.wigner_full.transpose(1, 2), ordered)


class CueqOperator(NativeOperator, nn.Module):
    def __init__(self, irreps_in, irreps_out, mmax, weights, geometry, *, method, rotation,
                 descriptor_name="escn_tp"):
        super().__init__()
        from operator_cueq import CueqLocal, cue_irreps
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        self.local = CueqLocal(irreps_in, irreps_out, mmax, method=method, device=weights[0].device,
                               descriptor_name=descriptor_name)
        self.weight = nn.Parameter(self.local.map_weights(weights).detach())
        self.rotation = rotation
        self.metadata = {"descriptor": descriptor_name, "method": self.local.method, "rotation": rotation,
                         "version": cue.__version__, "file": cue.__file__, "torch_file": cuet.__file__}
        if rotation == "cueq":
            self.rot_in = cuet.Rotation(cue_irreps(irreps_in), layout=cue.ir_mul,
                                        device=weights[0].device, math_dtype=torch.float32)
            self.rot_out = cuet.Rotation(cue_irreps(irreps_out), layout=cue.ir_mul,
                                         device=weights[0].device, math_dtype=torch.float32)
            self.metadata["rotation_methods"] = [self.rot_in.method, self.rot_out.method]
            self.input_layout = FeatureLayout(self.irreps_in, weights[0].device)
            self.output_layout = FeatureLayout(self.irreps_out, weights[0].device)
            self.alpha, self.beta = geometry.alpha, geometry.beta
            self.zero_angle = torch.zeros_like(self.alpha)
            if descriptor_name == "escn_tp_compact":
                from operator_cueq import compact_permutation
                self.register_buffer("compact_input", self.input_layout.inverse.index_select(
                    0, compact_permutation(self.irreps_in, weights[0].device)), persistent=False)
                output_order = self.output_layout.inverse.index_select(
                    0, compact_permutation(self.irreps_out, weights[0].device))
                self.register_buffer("compact_output_inverse", output_order.argsort(), persistent=False)
            self.metadata["feature_layout"] = "native ir_mul; cuet.Rotation has no layout transpose"
        else:
            self.compact_in = NativeWigner(self.irreps_in, geometry.blocks,
                                          compact=descriptor_name == "escn_tp_compact")
            self.compact_out = (self.compact_in if self.irreps_in == self.irreps_out else
                                NativeWigner(self.irreps_out, geometry.blocks,
                                             compact=descriptor_name == "escn_tp_compact"))
            self.input_layout, self.output_layout = self.compact_in.layout, self.compact_out.layout
            self.metadata["rotation_layout"] = "descriptor permutation merged into coefficient Wigner matrices"
            self.metadata["feature_layout"] = ("native [edges, coefficients, channels]" if
                self.compact_in.single_group and self.compact_out.single_group else
                "native flat coefficient/channel layout grouped by multiplicity; split once")

    def forward_native(self, x):
        if self.rotation != "cueq":
            return self.compact_out.rotate_inv(self.local.forward_native(self.compact_in.rotate(x), self.weight))
        local = self.rot_in(-self.alpha, -self.beta, self.zero_angle, x)
        if self.local.descriptor_name == "escn_tp_compact":
            local = local.index_select(-1, self.compact_input)
        out = self.local.forward_native(local, self.weight)
        if self.local.descriptor_name == "escn_tp_compact":
            out = out.index_select(-1, self.compact_output_inverse)
        return self.rot_out(self.zero_angle, self.beta, self.alpha, out)

    def canonical_gradients(self):
        return self.local.canonical_gradients(self.weight.grad)
