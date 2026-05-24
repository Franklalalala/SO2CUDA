from __future__ import annotations

from collections import OrderedDict

import torch

from so2_cuda_ops._scheduler_backend import (
    _PersistentGroupedP1Function as _SO2SchedulerAutogradFunction,
    _load_extension,
    _mainloop_kind,
    _prepare_route_layout,
    _wigner_tensor_and_mode,
)


class SO2CudaSchedulerFunction(_SO2SchedulerAutogradFunction):
    """Neutral SO2 scheduler facade over the native scheduler kernel.

    Callers depend on this module so the public Python path is about SO2
    schedule descriptors instead of model-specific route plumbing.
    """


def load_scheduler_extension():
    return _load_extension()


_SINGLE_ROUTE_LAYOUT_CACHE: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = OrderedDict()
_SINGLE_ROUTE_LAYOUT_CACHE_MAX = 64


def mainloop_kind(name: str | None = None) -> int:
    return _mainloop_kind(name)


def prepare_single_route_layout(
    graph_index: torch.Tensor,
    *,
    n_routes: int,
    n_problems: int,
    block_m: int,
    block_n: int,
    out_ptr: torch.Tensor,
    raw_pair_tiles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _prepare_route_layout(
        graph_index,
        int(n_routes),
        int(n_problems),
        int(block_m),
        int(block_n),
        out_ptr,
        raw_pair_tiles=bool(raw_pair_tiles),
    )


def prepare_so2_single_route_layout(
    *,
    num_rows: int,
    n_problems: int,
    block_m: int,
    block_n: int,
    out_ptr: torch.Tensor,
    raw_pair_tiles: bool,
    nosync: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (
        str(out_ptr.device),
        int(num_rows),
        int(n_problems),
        int(block_m),
        int(block_n),
        int(out_ptr.data_ptr()),
        int(out_ptr.numel()),
        int(getattr(out_ptr, "_version", 0)),
        bool(raw_pair_tiles),
        bool(nosync),
    )
    cached = _SINGLE_ROUTE_LAYOUT_CACHE.get(key)
    if cached is not None:
        _SINGLE_ROUTE_LAYOUT_CACHE.move_to_end(key)
        return cached

    if nosync:
        edge_order = torch.arange(int(num_rows), dtype=torch.long, device=out_ptr.device)
        route_ptr = torch.empty((2,), dtype=torch.long, device=out_ptr.device)
        route_ptr[0] = 0
        route_ptr[1] = int(num_rows)
        widths = out_ptr[1:] - out_ptr[:-1]
        if raw_pair_tiles:
            widths = widths * 2
        row_tiles = (int(num_rows) + int(block_m) - 1) // int(block_m)
        col_tiles = torch.div(widths + int(block_n) - 1, int(block_n), rounding_mode="floor")
        problem_tiles = col_tiles * int(row_tiles)
        problem_tile_prefix = torch.empty((int(n_problems) + 1,), dtype=torch.long, device=out_ptr.device)
        problem_tile_prefix[:1].zero_()
        if int(n_problems) > 0:
            problem_tile_prefix[1:] = torch.cumsum(problem_tiles, dim=0)
    else:
        ext = load_scheduler_extension()
        edge_order, route_ptr, problem_tile_prefix = ext.so2_single_route_layout(
            int(num_rows),
            int(n_problems),
            int(block_m),
            int(block_n),
            out_ptr,
            bool(raw_pair_tiles),
        )
    cached = edge_order.contiguous(), route_ptr.contiguous(), problem_tile_prefix.contiguous()
    _SINGLE_ROUTE_LAYOUT_CACHE[key] = cached
    while len(_SINGLE_ROUTE_LAYOUT_CACHE) > _SINGLE_ROUTE_LAYOUT_CACHE_MAX:
        _SINGLE_ROUTE_LAYOUT_CACHE.popitem(last=False)
    return cached


def wigner_tensor_and_mode(module, wigner_D_all, x: torch.Tensor):
    return _wigner_tensor_and_mode(module, wigner_D_all, x)


def materialized_scheduler(
    *,
    num_rows: int,
    n_problems: int,
    block_m: int,
    block_n: int,
    out_ptr: torch.Tensor,
    raw_pair_tiles: bool = False,
    nosync: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the single-route SO2 materialized scheduler descriptor."""

    return prepare_so2_single_route_layout(
        num_rows=num_rows,
        n_problems=n_problems,
        block_m=block_m,
        block_n=block_n,
        out_ptr=out_ptr,
        raw_pair_tiles=raw_pair_tiles,
        nosync=nosync,
    )
