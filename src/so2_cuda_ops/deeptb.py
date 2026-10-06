"""Tensor-only integration API for SO2 layers and routed linear operators.

Model construction, radial networks, expert parameterization and checkpoint state
belong to the caller. This module accepts only tensors and layout descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from ._permutation import permute_rows
from .grouped_gemm import grouped_gemm, grouped_gemm_multi


@dataclass
class WignerData:
    values: torch.Tensor
    compact_offsets: torch.Tensor
    mode: int
    stride: int


@dataclass
class PairLayout:
    """Per-m (input base, input l, output base, output l, l offsets) maps."""
    maps: tuple[tuple[torch.Tensor, ...], ...]
    out_dim: int
    rotate_in: bool = True
    rotate_out: bool = True
    front: bool = True
    activation_layout: dict | None = None


@dataclass
class LinearWeights:
    """Routed [experts, out, in] or interpolation [out, in] parameters."""
    weight: torch.Tensor
    bias: torch.Tensor | None = None
    shared_weight: torch.Tensor | None = None
    shared_bias: torch.Tensor | None = None
    routed: bool = True


@dataclass
class DenseRouting:
    graph_index: torch.Tensor
    ptr: torch.Tensor
    permute: torch.Tensor | None = None
    unpermute: torch.Tensor | None = None

    def indexed_flat_permutation(self, graph_index, pair):
        return self.permute, self.unpermute, graph_index

    def indexed_segment_ptr(self, sorted_graph_index, num_groups, *, prefer_cpu=True):
        return self.ptr


@dataclass
class ActivationRouting:
    indices: torch.Tensor
    values: torch.Tensor
    slots: tuple[tuple[torch.Tensor, ...], ...]
    coefficients_sum_to_one: bool = False
    branch: str = "all"

    @property
    def topk_indices(self):
        return self.indices

    @property
    def topk_values(self):
        return self.values

    def expert_slot_layout(self, slot, flat, num_experts):
        return self.slots[slot]


class _LinearView:
    def __init__(self, params):
        self.params = params
        self.in_features = params.weight.shape[-1]
        self.out_features = params.weight.shape[-2]
        self.num_experts = params.weight.shape[0] if params.routed else 0
        self.num_shared_experts = 0 if params.shared_weight is None else params.shared_weight.shape[0]
        self.weight_shared = params.shared_weight
        self.bias_shared = params.shared_bias
        self.bias_experts = params.bias

    def _mix_expert_parameters(self, routing):
        return self.params.weight, self.params.bias

    def __call__(self, x):
        return F.linear(x, self.params.weight, self.params.bias)


class _PairLinearView:
    def __init__(self, params):
        self.fc = _LinearView(params)
        self.is_mole = params.routed
        self.num_out_channel = params.weight.shape[-2] // 2

    @staticmethod
    def _finish_linear_output(raw):
        width = raw.shape[-1] // 2
        real, imag = raw[..., :width], raw[..., width:]
        return torch.cat((real.narrow(1, 0, 1) - imag.narrow(1, 1, 1),
                          real.narrow(1, 1, 1) + imag.narrow(1, 0, 1)), dim=1)


class _LayerView:
    """Private adapter for the arithmetic-preserving kernel implementation."""
    def __init__(self, layout, linears, device):
        self.m_max = len(layout.maps) - 1
        self.irreps_out = SimpleNamespace(dim=layout.out_dim)
        self.rotate_in = layout.rotate_in
        self.rotate_out = layout.rotate_out
        self.front = layout.front
        self.fc_m0 = _LinearView(linears[0])
        self.m_linear = [_PairLinearView(p) for p in linears[1:]]
        self._so2_moe_fused_p0_pair_maps = {(m, str(device)): maps for m, maps in enumerate(layout.maps)}
        # Pair maps are already validated/prepared by the model adapter.
        self._so2_sandwich_pair_maps = self._so2_moe_fused_p0_pair_maps
        self._so2_activation_pair_maps = self._so2_moe_fused_p0_pair_maps
        self._so2_activation_layout = layout.activation_layout if layout.activation_layout is not None else {}
        layout.activation_layout = self._so2_activation_layout


def _supported(x, wigner):
    return (x.is_cuda and x.dtype == torch.float32 and not wigner.values.requires_grad
            and not torch.is_autocast_enabled()
            and not getattr(torch._C, '_are_functorch_transforms_active', lambda: False)())


def dense_pairs(x: torch.Tensor, layout: PairLayout, wigner: WignerData,
                weights: tuple[torch.Tensor, ...], radial_parts: tuple[torch.Tensor, ...] | None,
                routing: DenseRouting, *, forward_mode: str | None = None):
    """Return ordered m>0 contributions, or None before any unsupported computation.

    The caller forms its reference m=0 output first, then adds these contributions
    in order. ``weights`` includes an unused m=0 placeholder to preserve m indexing.
    Pair weights are [groups, 2*Cout_m, Cin_m]. Geometry is constant.
    """
    if not _supported(x, wigner):
        return None
    from . import tensor_product as tp
    mode = forward_mode or tp._env('DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE', 'scalar')
    multi = ('indexed_sandwich_multi', 'cublas_multi_sandwich', 'route_m_sandwich',
             'indexed_sandwich_multi_grouped', 'cublas_multi_sandwich_grouped')
    if mode not in ('scalar',) + multi:
        return None
    linears = tuple(LinearWeights(weight) for weight in weights)
    module = _LayerView(layout, linears, x.device)
    args = (module, x, wigner.values, wigner.compact_offsets, wigner.mode, wigner.stride, routing)
    if mode in multi:
        return tp._fused_pairs_indexed_sandwich_multi(*args, radial_parts)
    outputs = []
    for m in range(1, module.m_max + 1):
        radial = None if radial_parts is None else radial_parts[m].unsqueeze(1)
        out = tp._fused_pair_contribution(module, m, x, wigner.values, wigner.compact_offsets,
                                          wigner.mode, wigner.stride, routing, radial)
        if out is None:
            return None
        outputs.append(out)
    return outputs


def activation_forward(x: torch.Tensor, layout: PairLayout, wigner: WignerData,
                       linears: tuple[LinearWeights, ...], radials: tuple[torch.Tensor, ...] | None,
                       routing: ActivationRouting, *, schedule: str = 'per_slot'):
    """Pack, apply per-slot grouped expert GEMMs, and scatter the complete output.

    Top-k probabilities retain their autograd graph. Parameters are already folded
    by the caller when coefficients_sum_to_one is true. Returns None on CPU,
    unsupported dtype, differentiable Wigner matrices, autocast or torch.func.
    """
    if not _supported(x, wigner):
        return None
    if schedule not in ('per_slot', 'expanded'):
        raise ValueError('schedule must be per_slot or expanded')
    from . import _activation as activation
    from . import tensor_product as tp
    module = _LayerView(layout, linears, x.device)
    params = [(view if p.routed else None, p.weight, p.bias)
              for view, p in zip([module.fc_m0] + [m.fc for m in module.m_linear], linears)]
    packed = activation._Packed(tp, module, x,
                                (wigner.values, wigner.compact_offsets, wigner.mode, wigner.stride), radials)
    return activation._fused_p0(module, x, packed, routing, params, schedule, grouped_gemm_multi).contiguous()


__all__ = ['WignerData', 'PairLayout', 'LinearWeights', 'DenseRouting', 'ActivationRouting',
           'dense_pairs', 'activation_forward', 'grouped_gemm', 'grouped_gemm_multi', 'permute_rows']


def prepare_wigner(x: torch.Tensor, values: torch.Tensor | tuple[torch.Tensor, ...] | None,
                   *, l_max: int, rotate: bool = True):
    """Pack dense [N,D,D] or compact per-l [N,2l+1,2l+1] Wigner data."""
    from . import tensor_product as tp
    from ._compat import SO2WignerBlocks
    if isinstance(values, (tuple, list)):
        values = SO2WignerBlocks(values)
    if tp._wigner_requires_grad(values):
        return None
    view = SimpleNamespace(rotate_in=rotate, rotate_out=rotate, l_max=l_max,
                           dims=tuple(2*l+1 for l in range(l_max+1)))
    packed = tp._wigner_tensor_and_mode(view, values, x)
    return None if packed is None else WignerData(*packed)


def prepare_layout(in_entries: tuple[tuple[int, int, int], ...],
                   out_entries: tuple[tuple[int, int, int], ...], *,
                   m_max: int, l_max: int, out_dim: int, device,
                   rotate_in: bool = True, rotate_out: bool = True, front: bool = True):
    """Build reusable integer maps from (l, multiplicity, first feature) entries."""
    maps = []
    with torch.inference_mode(False), torch.no_grad():
        offsets = torch.tensor([l*l for l in range(l_max+1)], dtype=torch.long, device=device)
        for m in range(m_max+1):
            columns = []
            for entries in (in_entries, out_entries):
                bases, levels = [], []
                for l, mul, start in entries:
                    if l >= m:
                        bases.extend(start + c*(2*l+1) for c in range(mul))
                        levels.extend([l]*mul)
                columns.extend((torch.tensor(bases, dtype=torch.long, device=device),
                                torch.tensor(levels, dtype=torch.long, device=device)))
            maps.append((*columns, offsets))
    return PairLayout(tuple(maps), out_dim, rotate_in, rotate_out, front)


__all__ += ['prepare_wigner', 'prepare_layout']
