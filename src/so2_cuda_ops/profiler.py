from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any


_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}
_TRUE = {"1", "true", "True", "TRUE", "on", "ON", "yes", "Yes"}
_STATE_LOCK = threading.Lock()
_CUDA_PENDING: list[tuple[str, Any, Any]] = []
_CUDA_STATS: dict[str, dict[str, float]] = {}
_HOST_STATS: dict[str, dict[str, float]] = {}
_PRINT_COUNTER = 0


def _env_any(names: tuple[str, ...], default: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return str(value)
    return str(default)


def _int_env_any(names: tuple[str, ...], default: int) -> int:
    value = _env_any(names, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _bool_env_any(names: tuple[str, ...], default: str = "0") -> bool:
    value = _env_any(names, default)
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return bool(value)


def profile_enabled() -> bool:
    return _bool_env_any(("SO2_CUDA_PROFILE", "DPTB_SO2_PROFILE"), "0")


def profile_detail_enabled() -> bool:
    return _bool_env_any(("SO2_CUDA_PROFILE_DETAIL", "DPTB_SO2_PROFILE_DETAIL"), "0")


def _stat_add(stats: dict[str, dict[str, float]], label: str, ms: float) -> None:
    row = stats.setdefault(label, {"count": 0.0, "total": 0.0, "max": 0.0})
    row["count"] += 1.0
    row["total"] += float(ms)
    row["max"] = max(row["max"], float(ms))


def _torch_cuda_device(device_like: Any):
    try:
        import torch
    except Exception:
        return None, None
    device = getattr(device_like, "device", device_like)
    try:
        device = torch.device(device)
    except Exception:
        try:
            device = torch.device("cuda")
        except Exception:
            return torch, None
    if device.type != "cuda" or not torch.cuda.is_available():
        return torch, None
    return torch, device


def cuda_span_start(label: str, device_like: Any = None):
    if not profile_enabled():
        return None
    torch, device = _torch_cuda_device(device_like)
    if torch is None or device is None:
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    start.record(stream)
    return (str(label), start, end, stream)


def cuda_span_end(token) -> None:
    if token is None:
        return
    label, start, end, stream = token
    end.record(stream)
    with _STATE_LOCK:
        _CUDA_PENDING.append((str(label), start, end))


def record_cuda_span(label: str, device_like: Any, fn: Callable[[], Any]):
    if not profile_enabled():
        return fn()
    torch, device = _torch_cuda_device(device_like)
    if torch is None or device is None:
        return fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    stream = torch.cuda.current_stream(device)
    start.record(stream)
    try:
        return fn()
    finally:
        end.record(stream)
        with _STATE_LOCK:
            _CUDA_PENDING.append((str(label), start, end))


def record_host_span(label: str, fn: Callable[[], Any]):
    if not profile_enabled():
        return fn()
    t0 = time.perf_counter()
    try:
        return fn()
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        with _STATE_LOCK:
            _stat_add(_HOST_STATS, str(label), elapsed_ms)


def _flush_cuda_locked() -> None:
    pending = list(_CUDA_PENDING)
    _CUDA_PENDING.clear()
    for label, start, end in pending:
        end.synchronize()
        _stat_add(_CUDA_STATS, label, float(start.elapsed_time(end)))


def _copy_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    copied: dict[str, dict[str, float]] = {}
    for label, row in stats.items():
        count = max(1.0, row["count"])
        copied[label] = {
            "count": int(row["count"]),
            "total": float(row["total"]),
            "mean": float(row["total"] / count),
            "max": float(row["max"]),
        }
    return copied


def _derived_cuda_stats(cuda_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    total = cuda_stats.get("so2.forward_total")
    if not total:
        return {}
    known_total = 0.0
    known_count = 0
    for label, row in cuda_stats.items():
        if label == "so2.forward_total":
            continue
        if label.startswith("so2.forward."):
            known_total += float(row["total"])
            known_count += int(row["count"])
    remaining = max(0.0, float(total["total"]) - known_total)
    count = max(1, int(total["count"]))
    return {
        "so2.forward_unattributed": {
            "count": count,
            "total": remaining,
            "mean": remaining / count,
            "max": remaining,
            "known_child_count": known_count,
        }
    }


def get_profile_summary(*, reset: bool = False) -> dict[str, Any]:
    with _STATE_LOCK:
        _flush_cuda_locked()
        cuda = _copy_stats(_CUDA_STATS)
        host = _copy_stats(_HOST_STATS)
        derived = _derived_cuda_stats(cuda)
        if reset:
            _CUDA_STATS.clear()
            _HOST_STATS.clear()
    return {
        "enabled": profile_enabled(),
        "cuda_ms": cuda,
        "derived_cuda_ms": derived,
        "host_ms": host,
    }


def reset_profile_summary() -> None:
    with _STATE_LOCK:
        _CUDA_PENDING.clear()
        _CUDA_STATS.clear()
        _HOST_STATS.clear()


def maybe_print_profile(context: str = "") -> None:
    global _PRINT_COUNTER
    if not profile_enabled():
        return
    every = _int_env_any(("SO2_CUDA_PROFILE_PRINT_EVERY", "DPTB_SO2_PROFILE_PRINT_EVERY"), 0)
    if every <= 0:
        return
    _PRINT_COUNTER += 1
    if _PRINT_COUNTER % every != 0:
        return
    summary = get_profile_summary(reset=False)
    rows = []
    for kind in ("cuda_ms", "derived_cuda_ms", "host_ms"):
        for label, row in sorted(summary[kind].items()):
            rows.append(f"{kind}:{label}=mean:{row['mean']:.4f} total:{row['total']:.4f} count:{row['count']}")
    prefix = "[SO2CUDA profile]"
    if context:
        prefix += f" {context}"
    print(prefix + " " + " | ".join(rows), file=sys.stderr, flush=True)
