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
        # Layout-derived index maps are cached on the persistent layout, not on this per-call view.
        self._cache_owner = layout


def _supported(x, wigner):
    return (x.is_cuda and x.dtype == torch.float32 and not wigner.values.requires_grad
            and not torch.is_autocast_enabled()
            and not getattr(torch._C, '_are_functorch_transforms_active', lambda: False)())


def _single_group(x, layout, weights, routing):
    """True when one routed group covers all 2N pair rows in their own order and
    every m>0 weight is [1, 2*Cout_m, Cin_m] on the input's device and dtype.

    Only a host-side pointer is inspected, so the check never synchronizes."""
    ptr = routing.ptr
    if routing.permute is not None or ptr.is_cuda or ptr.numel() != 2 or len(weights) != len(layout.maps):
        return False
    for m, w in enumerate(weights[1:], 1):
        cin, cout = layout.maps[m][0].numel(), layout.maps[m][2].numel()
        if w.shape != (1, 2 * cout, cin) or w.device != x.device or w.dtype != x.dtype:
            return False
    first, last = (int(v) for v in ptr.tolist())
    return first == 0 and last == 2 * x.shape[0]


def _edge_grouping(x, layout, weights, routing):
    """(ptr over edges, order, rows) of a multi-group route over every pair row, or None.

    The caller sorts pair rows by graph index and gives the CPU pointer over pair
    rows; both rows of an edge share its group, so the pointer over edges is half of
    it. The edge order is recomputed here as a stable sort of the graph index, so no
    property of the caller's pair-row permutation is assumed."""
    ptr = routing.ptr
    if ptr.is_cuda or ptr.ndim != 1 or ptr.numel() < 2 or len(weights) != len(layout.maps):
        return None
    groups = ptr.numel() - 1
    for m, w in enumerate(weights[1:], 1):
        cin, cout = layout.maps[m][0].numel(), layout.maps[m][2].numel()
        if w.shape != (groups, 2 * cout, cin) or w.device != x.device or w.dtype != x.dtype:
            return None
    bounds = [int(v) for v in ptr.tolist()]
    if (bounds[0] != 0 or bounds[-1] != 2 * x.shape[0] or any(b % 2 for b in bounds)
            or any(b > c for b, c in zip(bounds, bounds[1:]))):
        return None
    ptr_edges = torch.tensor([b // 2 for b in bounds], dtype=torch.long)
    if routing.permute is None:
        return ptr_edges, None, None
    graph = routing.graph_index
    if graph.ndim != 1 or graph.numel() != x.shape[0] or graph.device != x.device:
        return None
    order = torch.argsort(graph.to(torch.long), stable=True)
    rows = torch.empty_like(order)
    rows.scatter_(0, order, torch.arange(order.numel(), device=order.device, dtype=order.dtype))
    return ptr_edges, order, rows


def _radials_fit(x, layout, radial_parts, first_m=1):
    """No radial weights, or one radial weight per m block: [N, Cin_m] for front
    layouts, [N, Cout_m] otherwise."""
    if radial_parts is None:
        return True
    if len(radial_parts) < len(layout.maps):
        return False
    side = 0 if layout.front else 2
    return all(radial_parts[m].shape == (x.shape[0], layout.maps[m][side].numel())
               and radial_parts[m].device == x.device and radial_parts[m].dtype == x.dtype
               for m in range(first_m, len(layout.maps)))


def _dense_with_m0(x, layout, wigner, weights, radial_parts, routing):
    """[whole layer output] of dense_pairs(include_m0=True), or None."""
    if len(weights) != len(layout.maps) or not _radials_fit(x, layout, radial_parts, first_m=0):
        return None
    w0 = weights[0]
    cin0, cout0 = layout.maps[0][0].numel(), layout.maps[0][2].numel()
    if w0.ndim != 3 or w0.shape[1:] != (cout0, cin0) or w0.device != x.device or w0.dtype != x.dtype:
        return None
    from ._sandwich import _slot, full_sandwich, sandwich_plan
    if not sandwich_plan(layout, x.shape[1], x.device, with_m0=True).supported:
        return None
    if w0.shape[0] == 1 and _single_group(x, layout, weights, routing):
        out = full_sandwich(x, layout, wigner, w0[0], None, (None,) + tuple(w[0] for w in weights[1:]),
                            radial_parts)
    else:
        grouping = _edge_grouping(x, layout, weights, routing)
        if grouping is None or w0.shape[0] != grouping[0].numel() - 1:
            return None
        ptr_edges, order, rows = grouping
        out = full_sandwich(x, layout, wigner, w0, None, weights, radial_parts,
                            slots=[_slot(rows, order, ptr_edges, x.device)])
    return None if out is None else [out]


def dense_pairs(x: torch.Tensor, layout: PairLayout, wigner: WignerData,
                weights: tuple[torch.Tensor, ...], radial_parts: tuple[torch.Tensor, ...] | None,
                routing: DenseRouting, *, forward_mode: str | None = None, include_m0: bool = False):
    """Return ordered m>0 contributions, or None before any unsupported computation.

    The caller forms its reference m=0 output first, then adds these contributions
    in order. ``weights`` includes an unused m=0 placeholder to preserve m indexing.
    Pair weights are [groups, 2*Cout_m, Cin_m]. Geometry is constant.

    With ``include_m0=True``, ``weights[0]`` is the [groups, Cout_0, Cin_0] m=0
    weight and ``radial_parts[0]`` its radial weight; the m=0 term is computed with
    the pairs and the single returned tensor is the whole layer output.
    """
    if not _supported(x, wigner):
        return None
    from . import tensor_product as tp
    mode = forward_mode or tp._env('DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE', tp.DEFAULT_FUSED_P0_FORWARD_MODE)
    multi = ('indexed_sandwich_multi', 'cublas_multi_sandwich', 'route_m_sandwich',
             'indexed_sandwich_multi_grouped', 'cublas_multi_sandwich_grouped')
    if mode not in ('scalar',) + multi:
        return None
    if include_m0:
        return _dense_with_m0(x, layout, wigner, weights, radial_parts, routing)
    if mode in multi and _radials_fit(x, layout, radial_parts):
        from ._sandwich import block_sandwich, grouped_sandwich, sandwich_plan
        if _single_group(x, layout, weights, routing):
            # One group over every row: each m block is one plain GEMM of its pair rows.
            if sandwich_plan(layout, x.shape[1], x.device).supported:
                return [block_sandwich(x, layout, wigner, (None,) + tuple(w[0] for w in weights[1:]),
                                       radial_parts)]
        else:
            grouping = _edge_grouping(x, layout, weights, routing)
            if grouping is not None and sandwich_plan(layout, x.shape[1], x.device).supported:
                return [grouped_sandwich(x, layout, wigner, weights, radial_parts, *grouping)]
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


def _single_expert_forward(x, layout, wigner, linears, radials, routing):
    """One routed expert per block, top-1 and no separate shared term: the layer is
    gate * (D^T W D x + b) with one weight per block, computed by one block sandwich
    with the gate as a per-edge output scale. Returns None for every other case."""
    if routing.indices.ndim != 2 or routing.indices.shape[1] != 1 or routing.branch != 'all':
        return None
    if len(linears) != len(layout.maps) or len(layout.maps) < 1:
        return None
    fold = bool(routing.coefficients_sum_to_one)
    for m, params in enumerate(linears):
        cin, cout = layout.maps[m][0].numel(), layout.maps[m][2].numel()
        weight = params.weight
        if (not params.routed or weight.ndim != 3 or weight.shape != (1, cout * (2 if m else 1), cin)
                or weight.device != x.device or weight.dtype != x.dtype):
            return None
        if params.bias is not None and (m > 0 or params.bias.shape != (1, cout)):
            return None
        if params.shared_weight is not None and not fold:
            return None
    if layout.maps[0][0].numel() == 0 or layout.maps[0][2].numel() == 0:
        return None
    radials_by_m = _routed_radials(x, layout, radials)
    if radials_by_m is False:
        return None
    from ._sandwich import full_sandwich, sandwich_plan
    if not sandwich_plan(layout, x.shape[1], x.device, with_m0=True).supported:
        return None
    gate = routing.values.to(device=x.device, dtype=x.dtype)[:, 0]
    w0 = linears[0].weight[0]
    b0 = None if linears[0].bias is None else linears[0].bias[0]
    pairs = (None,) + tuple(params.weight[0] for params in linears[1:])
    return full_sandwich(x, layout, wigner, w0, b0, pairs, radials_by_m, gates=(gate,))


def _routed_radials(x, layout, radials):
    """Per-block radial weights of an activation layer, or False when they do not fit."""
    if radials is None:
        return None
    if layout.front and len(layout.maps) > 1:
        if len(radials) != 2:
            return False
        sizes = [layout.maps[m][0].numel() for m in range(1, len(layout.maps))]
        if radials[1].shape != (x.shape[0], sum(sizes)):
            return False
        radials_by_m = (radials[0],) + tuple(torch.split(radials[1], sizes, dim=-1))
    else:
        radials_by_m = tuple(radials)
    return radials_by_m if _radials_fit(x, layout, radials_by_m, first_m=0) else False


def _routed_forward(x, layout, wigner, linears, radials, routing, schedule):
    """Top-k routed experts, every block routed, no separate shared term: per slot,
    rotate the input into expert-sorted block rows, one grouped GEMM per block over
    the experts' row segments, and add gate * (rotated-back output). Returns None for
    every other case."""
    if schedule != 'per_slot' or routing.branch != 'all':
        return None
    idx = routing.indices
    n = x.shape[0]
    if idx.ndim != 2 or idx.shape[0] != n or len(routing.slots) != idx.shape[1] or len(linears) != len(layout.maps):
        return None
    experts = int(linears[0].weight.shape[0]) if linears[0].weight.ndim == 3 else 0
    fold = bool(routing.coefficients_sum_to_one)
    for m, params in enumerate(linears):
        cin, cout = layout.maps[m][0].numel(), layout.maps[m][2].numel()
        weight = params.weight
        if (not params.routed or weight.ndim != 3 or weight.shape != (experts, cout * (2 if m else 1), cin)
                or weight.device != x.device or weight.dtype != x.dtype):
            return None
        if params.bias is not None and (m > 0 or params.bias.shape != (experts, cout)):
            return None
        if params.shared_weight is not None and not fold:
            return None
    if experts == 0 or layout.maps[0][0].numel() == 0 or layout.maps[0][2].numel() == 0:
        return None
    radials_by_m = _routed_radials(x, layout, radials)
    if radials_by_m is False:
        return None
    from ._sandwich import _slot, full_sandwich, sandwich_plan
    slots = []
    for order, inverse, ptr, *rest in routing.slots:
        if (ptr.is_cuda or ptr.ndim != 1 or ptr.numel() != experts + 1 or int(ptr[0]) != 0
                or int(ptr[-1]) != n or order.numel() != n or inverse.numel() != n):
            return None
        # The sorted expert ids of the slot are the group of every sorted row; using them
        # avoids expanding the CPU pointer on the device (a synchronizing copy per layer).
        sorted_ids = rest[0] if rest else None
        if (sorted_ids is None or not torch.is_tensor(sorted_ids) or sorted_ids.shape != (n,)
                or sorted_ids.device != x.device):
            sorted_ids = None
        slots.append(_slot(inverse.to(device=x.device, dtype=torch.long),
                           order.to(device=x.device, dtype=torch.long), ptr.to(torch.long), x.device,
                           group_rows=None if sorted_ids is None else sorted_ids.to(torch.long)))
    if not sandwich_plan(layout, x.shape[1], x.device, with_m0=True).supported:
        return None
    values = routing.values.to(device=x.device, dtype=x.dtype)
    gates = tuple(values[:, j] for j in range(values.shape[1]))
    pairs = (None,) + tuple(params.weight for params in linears[1:])
    return full_sandwich(x, layout, wigner, linears[0].weight, linears[0].bias, pairs, radials_by_m,
                         gates=gates, slots=slots)


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
    fused = _single_expert_forward(x, layout, wigner, linears, radials, routing)
    if fused is None:
        fused = _routed_forward(x, layout, wigner, linears, radials, routing, schedule)
    if fused is not None:
        return fused
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

from ._dense import true_dense_pairs

__all__ += ['true_dense_pairs']
