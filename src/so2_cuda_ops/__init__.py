from __future__ import annotations

from ._version import __version__
from .backend import is_available
from .config import BackendConfig, get_backend_config, set_backend_config
from .profiler import get_profile_summary, profile_enabled, reset_profile_summary


def grouped_gemm(*args, **kwargs):
    from .grouped_gemm import grouped_gemm as _grouped_gemm

    return _grouped_gemm(*args, **kwargs)


def grouped_gemm_multi(*args, **kwargs):
    from .grouped_gemm import grouped_gemm_multi as _grouped_gemm_multi

    return _grouped_gemm_multi(*args, **kwargs)


def indexed_sandwich_multi_gemm(*args, **kwargs):
    from .grouped_gemm import indexed_sandwich_multi_gemm as _indexed_sandwich_multi_gemm

    return _indexed_sandwich_multi_gemm(*args, **kwargs)


def indexed_sandwich_multi_block_gemm(*args, **kwargs):
    from .grouped_gemm import indexed_sandwich_multi_block_gemm as _indexed_sandwich_multi_block_gemm

    return _indexed_sandwich_multi_block_gemm(*args, **kwargs)


def indexed_sandwich_multi_block_direct_gemm(*args, **kwargs):
    from .grouped_gemm import indexed_sandwich_multi_block_direct_gemm as _indexed_sandwich_multi_block_direct_gemm

    return _indexed_sandwich_multi_block_direct_gemm(*args, **kwargs)


def indexed_sandwich_multi(*args, **kwargs):
    return indexed_sandwich_multi_gemm(*args, **kwargs)


__all__ = [
    "BackendConfig",
    "__version__",
    "get_backend_config",
    "get_profile_summary",
    "grouped_gemm",
    "grouped_gemm_multi",
    "indexed_sandwich_multi",
    "indexed_sandwich_multi_block_direct_gemm",
    "indexed_sandwich_multi_block_gemm",
    "indexed_sandwich_multi_gemm",
    "is_available",
    "profile_enabled",
    "reset_profile_summary",
    "set_backend_config",
]
