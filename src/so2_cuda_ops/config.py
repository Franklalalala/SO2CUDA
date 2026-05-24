from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}


@dataclass(frozen=True)
class BackendConfig:
    backend: str = "auto"
    min_edges: int = 0
    materialized_min_edges: int = 0
    gemm_strategy: str = "scheduler"
    fast_tf32: bool = False


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def get_backend_config() -> BackendConfig:
    return BackendConfig(
        backend=os.environ.get("SO2_CUDA_BACKEND", "auto"),
        min_edges=_int_env("SO2_CUDA_MIN_EDGES", 0),
        materialized_min_edges=_int_env("SO2_CUDA_MATERIALIZED_MIN_EDGES", 0),
        gemm_strategy=os.environ.get("SO2_CUDA_GEMM_STRATEGY", "scheduler"),
        fast_tf32=_bool_env("SO2_CUDA_FAST_TF32", "0"),
    )


def set_backend_config(
    *,
    backend: Optional[str] = None,
    min_edges: Optional[int] = None,
    materialized_min_edges: Optional[int] = None,
    gemm_strategy: Optional[str] = None,
    fast_tf32: Optional[bool] = None,
) -> BackendConfig:
    if backend is not None:
        os.environ["SO2_CUDA_BACKEND"] = str(backend)
    if min_edges is not None:
        os.environ["SO2_CUDA_MIN_EDGES"] = str(int(min_edges))
    if materialized_min_edges is not None:
        os.environ["SO2_CUDA_MATERIALIZED_MIN_EDGES"] = str(int(materialized_min_edges))
    if gemm_strategy is not None:
        os.environ["SO2_CUDA_GEMM_STRATEGY"] = str(gemm_strategy)
    if fast_tf32 is not None:
        os.environ["SO2_CUDA_FAST_TF32"] = "1" if fast_tf32 else "0"
    sync_legacy_env_aliases()
    return get_backend_config()


def sync_legacy_env_aliases() -> None:
    """Bridge new public env names to historical backend knobs when unset."""

    aliases = {
        "SO2_CUDA_FAST_TF32": "DPTB_CUBLAS_GROUPED_FAST_TF32",
        "SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR": "DPTB_CUBLAS_GROUPED_BUILD_DIR",
        "SO2_CUDA_PACK_SCATTER_BUILD_DIR": "DPTB_SO2_MOE_FUSED_P0_BUILD_DIR",
        "SO2_CUDA_PACK_SCATTER_VERBOSE": "DPTB_SO2_MOE_FUSED_P0_VERBOSE",
        "SO2_CUDA_SCHEDULER_BUILD_DIR": "DPTB_SO2_MOE_PERSISTENT_P1_BUILD_DIR",
        "SO2_CUDA_SCHEDULER_VERBOSE": "DPTB_SO2_MOE_PERSISTENT_P1_VERBOSE",
        "SO2_CUDA_CUTLASS_ROOT": "DPTB_CUTLASS_ROOT",
    }
    for public, legacy in aliases.items():
        if public in os.environ and legacy not in os.environ:
            os.environ[legacy] = os.environ[public]
