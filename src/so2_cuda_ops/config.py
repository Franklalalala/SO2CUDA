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
    """Bridge public SO2CUDA env names to historical DeePTB aliases.

    New code and documentation should use ``SO2_CUDA_*``. The legacy names are
    populated only so migrated DeePTB call paths can keep working while their
    internal wrappers are thin adapters.
    """

    aliases = {
        "SO2_CUDA_FAST_TF32": ("DPTB_CUBLAS_GROUPED_FAST_TF32",),
        "SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR": ("DPTB_CUBLAS_GROUPED_BUILD_DIR",),
        "SO2_CUDA_CUBLAS_GROUPED_VERBOSE": ("DPTB_CUBLAS_GROUPED_VERBOSE",),
        "SO2_CUDA_CUTLASS_GROUPED_BUILD_DIR": ("DPTB_CUTLASS_GROUPED_BUILD_DIR",),
        "SO2_CUDA_CUTLASS_GROUPED_VERBOSE": ("DPTB_CUTLASS_GROUPED_VERBOSE",),
        "SO2_CUDA_CUTLASS_GEMM_SMOKE_BUILD_DIR": ("DPTB_CUTLASS_SO2_GEMM_SMOKE_BUILD_DIR",),
        "SO2_CUDA_CUTLASS_GEMM_SMOKE_VERBOSE": ("DPTB_CUTLASS_SO2_GEMM_SMOKE_VERBOSE",),
        "SO2_CUDA_PACK_SCATTER_BUILD_DIR": ("DPTB_SO2_MOE_FUSED_P0_BUILD_DIR",),
        "SO2_CUDA_PACK_SCATTER_VERBOSE": ("DPTB_SO2_MOE_FUSED_P0_VERBOSE",),
        "SO2_CUDA_SCHEDULER_BUILD_DIR": ("DPTB_SO2_MOE_PERSISTENT_P1_BUILD_DIR",),
        "SO2_CUDA_SCHEDULER_VERBOSE": ("DPTB_SO2_MOE_PERSISTENT_P1_VERBOSE",),
        "SO2_CUDA_CUTLASS_ROOT": (
            "DPTB_CUTLASS_ROOT",
            "DPTB_SO2_MOE_FUSED_P0_CUTLASS_ROOT",
            "DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT",
        ),
        "SO2_CUDA_LINEINFO": (
            "DPTB_CUTLASS_GROUPED_LINEINFO",
            "DPTB_CUTLASS_SO2_GEMM_SMOKE_LINEINFO",
            "DPTB_SO2_MOE_FUSED_P0_LINEINFO",
            "DPTB_SO2_MOE_PERSISTENT_P1_LINEINFO",
        ),
        "SO2_CUDA_FORWARD_MODE": ("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE",),
        "SO2_CUDA_BACKWARD_MODE": (
            "DPTB_SO2_MOE_FUSED_P0_BACKWARD_MODE",
            "DPTB_SO2_MOE_PERSISTENT_P1_BACKWARD_MODE",
        ),
        "SO2_CUDA_STRICT_FORWARD_MODE": ("DPTB_SO2_MOE_FUSED_P0_STRICT_FORWARD_MODE",),
        "SO2_CUDA_ASSUME_SORTED": (
            "DPTB_SO2_MOE_FUSED_P0_ASSUME_SORTED",
            "DPTB_SO2_MOE_PERSISTENT_P1_ASSUME_SORTED",
        ),
        "SO2_CUDA_LOG_ONCE": (
            "DPTB_SO2_MOE_FUSED_P0_LOG_ONCE",
            "DPTB_SO2_MOE_PERSISTENT_P1_LOG_ONCE",
        ),
        "SO2_CUDA_LOG_SCHEDULE": ("DPTB_SO2_MOE_FUSED_P0_LOG_SCHEDULE",),
        "SO2_CUDA_SCHEDULER_MAINLOOP": ("DPTB_SO2_MOE_PERSISTENT_P1_MAINLOOP",),
        "SO2_CUDA_SCHEDULER_BLOCK_M": ("DPTB_SO2_MOE_PERSISTENT_P1_BLOCK_M",),
        "SO2_CUDA_SCHEDULER_BLOCK_N": ("DPTB_SO2_MOE_PERSISTENT_P1_BLOCK_N",),
        "SO2_CUDA_SCHEDULER_ACTIVE_BLOCKS": ("DPTB_SO2_MOE_PERSISTENT_P1_ACTIVE_BLOCKS",),
        "SO2_CUDA_SCHEDULER_INCLUDE_M0": ("DPTB_SO2_MOE_PERSISTENT_P1_INCLUDE_M0",),
        "SO2_CUDA_SCHEDULER_NOSYNC_LAYOUT": ("DPTB_SO2_MOE_PERSISTENT_P1_NOSYNC_LAYOUT",),
        "SO2_CUDA_SCHEDULER_VALIDATE_ROUTE_IDS": ("DPTB_SO2_MOE_PERSISTENT_P1_VALIDATE_ROUTE_IDS",),
        "SO2_CUDA_SCHEDULER_CUTLASS_TILE": ("DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_TILE",),
    }
    for public, legacy_names in aliases.items():
        if public not in os.environ:
            continue
        for legacy in legacy_names:
            if legacy not in os.environ:
                os.environ[legacy] = os.environ[public]
