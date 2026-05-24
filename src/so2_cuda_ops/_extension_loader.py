from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch.utils.cpp_extension as torch_cpp_extension
from torch.utils.cpp_extension import load

_FALSE = {"", "0", "false", "False", "FALSE", "off", "OFF", "no", "No"}


def truthy_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in _FALSE


def _as_path_strings(paths: Optional[Iterable[str | Path]]) -> list[str]:
    if not paths:
        return []
    return [str(Path(path)) for path in paths]


def _candidate_toolkit_roots() -> list[Path]:
    roots: list[Path] = [Path(root) for root in (os.environ.get("CUDA_HOME"), os.environ.get("CUDA_PATH")) if root]
    roots.append(Path("/usr/local/cuda"))
    usr_local = Path("/usr/local")
    if usr_local.is_dir():
        roots.extend(sorted(usr_local.glob("cuda-*"), reverse=True))

    deduped = []
    seen = set()
    for root in roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        deduped.append(root)
    return deduped


def _ensure_cuda_toolkit() -> None:
    current = Path(str(getattr(torch_cpp_extension, "CUDA_HOME", "") or os.environ.get("CUDA_HOME", "")))
    current_bin = current / "bin"
    if (current_bin / "nvcc").is_file() and (current_bin / "cudafe++").is_file():
        return

    for root in _candidate_toolkit_roots():
        bin_dir = root / "bin"
        if not ((bin_dir / "nvcc").is_file() and (bin_dir / "cudafe++").is_file()):
            continue
        os.environ["CUDA_HOME"] = str(root)
        os.environ["CUDA_PATH"] = str(root)
        path = os.environ.get("PATH", "")
        bin_str = str(bin_dir)
        if bin_str not in path.split(os.pathsep):
            os.environ["PATH"] = bin_str + (os.pathsep + path if path else "")
        torch_cpp_extension.CUDA_HOME = str(root)
        return


def _env_cuda_paths() -> tuple[list[str], list[str]]:
    include_paths = []
    library_paths = []
    for site_root in site.getsitepackages():
        root = Path(site_root) / "nvidia"
        for package in ("cuda_runtime", "cublas"):
            include_dir = root / package / "include"
            library_dir = root / package / "lib"
            if include_dir.is_dir():
                include_paths.append(str(include_dir))
            if library_dir.is_dir():
                library_paths.append(str(library_dir))
    seen = set(include_paths)
    lib_seen = set(library_paths)
    for root in _candidate_toolkit_roots():
        for include_dir in (root / "include", root / "targets" / "x86_64-linux" / "include"):
            if (include_dir / "crt" / "host_config.h").is_file() and str(include_dir) not in seen:
                include_paths.append(str(include_dir))
                seen.add(str(include_dir))
        for library_dir in (root / "lib64", root / "targets" / "x86_64-linux" / "lib"):
            if library_dir.is_dir() and str(library_dir) not in lib_seen:
                library_paths.append(str(library_dir))
                lib_seen.add(str(library_dir))
    return include_paths, library_paths


def load_cuda_extension(
    *,
    name: str,
    source_files: Sequence[str | Path],
    build_dir_env: str,
    default_build_dir: str | Path,
    extra_cflags: Optional[list[str]] = None,
    extra_cuda_cflags: Optional[list[str]] = None,
    extra_ldflags: Optional[list[str]] = None,
    extra_include_paths: Optional[Iterable[str | Path]] = None,
    verbose_env: Optional[str] = None,
):
    """Load one CUDA extension using the repo's standard build conventions."""

    _ensure_cuda_toolkit()
    build_dir = Path(os.environ.get(build_dir_env, str(default_build_dir)))
    build_dir.mkdir(parents=True, exist_ok=True)
    env_include_paths, env_library_paths = _env_cuda_paths()
    include_paths = env_include_paths + _as_path_strings(extra_include_paths)
    ldflags = [f"-L{path}" for path in env_library_paths] + (extra_ldflags or [])
    return load(
        name=name,
        sources=_as_path_strings(source_files),
        extra_cflags=extra_cflags or ["-O3"],
        extra_cuda_cflags=extra_cuda_cflags or ["-O3"],
        extra_ldflags=ldflags,
        extra_include_paths=include_paths,
        build_directory=str(build_dir),
        with_cuda=True,
        verbose=truthy_env(verbose_env, "0") if verbose_env else False,
    )
