from __future__ import annotations

from typing import Any

import torch


class SO2WignerBlocks:
    """Small compact-Wigner protocol used by the SO2 CUDA kernels."""

    __slots__ = ("blocks",)

    def __init__(self, blocks):
        self.blocks = tuple(blocks)

    def block(self, l: int):
        return self.blocks[l]


def is_wigner_blocks(value) -> bool:
    return hasattr(value, "blocks") and callable(getattr(value, "block", None))


MOLEGlobals = Any


def _mole_split_sizes(mole_globals, n_rows: int):
    split_sizes = getattr(mole_globals, "split_sizes", None)
    if split_sizes is None:
        sizes_tensor = getattr(mole_globals, "_sizes_tensor", None)
        if sizes_tensor is None:
            split_sizes = (n_rows,)
        else:
            split_sizes = tuple(int(v) for v in sizes_tensor.detach().cpu().reshape(-1).tolist())
    if sum(split_sizes) != n_rows:
        raise ValueError(f"MOLE split sizes sum to {sum(split_sizes)}, but input has {n_rows} rows.")
    return split_sizes


def _mole_graph_index(mole_globals, n_rows: int, *, device):
    graph_index = getattr(mole_globals, "graph_index", None)
    if graph_index is not None:
        graph_index = graph_index.to(device=device, dtype=torch.long).reshape(-1)
        if graph_index.numel() == n_rows:
            return graph_index
        raise ValueError(f"MOLE graph_index has {graph_index.numel()} rows, but input has {n_rows} rows.")

    sizes_tensor = getattr(mole_globals, "_sizes_tensor", None)
    if sizes_tensor is not None:
        cache = getattr(mole_globals, "_graph_index_cache", None)
        if cache is None:
            cache = {}
            setattr(mole_globals, "_graph_index_cache", cache)
        key = (str(device), "tensor_sizes", int(sizes_tensor.numel()))
        cached = cache.get(key)
        if cached is None:
            sizes = sizes_tensor.to(device=device, dtype=torch.long)
            cached = torch.repeat_interleave(
                torch.arange(sizes.shape[0], dtype=torch.long, device=device),
                sizes,
                output_size=n_rows,
            )
            cache[key] = cached
        if cached.numel() == n_rows:
            return cached
        raise ValueError(f"MOLE sizes expand to {cached.numel()} rows, but input has {n_rows} rows.")

    split_sizes = _mole_split_sizes(mole_globals, n_rows)
    cache = getattr(mole_globals, "_graph_index_cache", None)
    if cache is None:
        cache = {}
        setattr(mole_globals, "_graph_index_cache", cache)
    key = (str(device), split_sizes)
    cached = cache.get(key)
    if cached is None:
        sizes = torch.tensor(split_sizes, dtype=torch.long, device=device)
        cached = torch.repeat_interleave(
            torch.arange(len(split_sizes), dtype=torch.long, device=device),
            sizes,
            output_size=n_rows,
        )
        cache[key] = cached
    return cached
