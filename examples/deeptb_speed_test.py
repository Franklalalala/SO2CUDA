#!/usr/bin/env python3
"""Compare SO2CUDA and PyTorch on randomly initialized UniTB dense/X1 models."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import functools
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("dense", "x1", "both"), default="both")
    parser.add_argument("--backend", choices=("reference", "cuda", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--edges", type=int, default=20000, help="Target directed edges, rounded up to complete cells")
    parser.add_argument("--side", type=int, default=5, help="Atoms along each axis of a periodic cubic cell")
    parser.add_argument("--neighbors", type=int, default=80, help="Even number of inversion-paired neighbors per atom")
    parser.add_argument("--spacing", type=float, default=2.1, help="Lattice spacing in angstrom")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--forward-mode", choices=("auto", "scalar", "indexed_sandwich_multi"), default="auto")
    parser.add_argument("--max-memory-gib", type=float, default=20.0, help="CUDA allocator limit; reduce --edges on OOM")
    parser.add_argument("--shared-gpu", action="store_true", help="Record that other workloads share the GPU")
    parser.add_argument("--json", type=Path, help="Write measurements and numerical differences as JSON")
    args = parser.parse_args()
    if (min(args.edges, args.side, args.neighbors, args.iterations, args.threads) <= 0
            or args.neighbors % 2 or args.side < 2 or args.warmup < 0
            or args.spacing <= 0 or args.max_memory_gib <= 0):
        parser.error("Positive sizes, side >= 2, even neighbors and warmup >= 0 are required")
    if args.threads > 8:
        parser.error("Use at most 8 CPU threads")
    return args


def install_counters():
    """Wrap the public adapter before DeePTB imports its entry points."""
    try:
        adapter = importlib.import_module("so2_cuda_ops.deeptb")
    except ImportError as exc:
        raise RuntimeError("Install SO2CUDA with its DeePTB adapter before using --backend cuda/both") from exc
    counts = {}
    for name in ("dense_pairs", "activation_forward", "grouped_gemm", "grouped_gemm_multi", "permute_rows"):
        original = getattr(adapter, name, None)
        if original is None:
            continue

        def counted(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            if result is not None:
                counts[_name] = counts.get(_name, 0) + 1
            return result

        setattr(adapter, name, functools.wraps(original)(counted))
    return counts


def neighbor_offsets(count):
    radius = max(1, math.ceil((count / 2) ** (1 / 3)))
    while True:
        pairs = []
        for offset in itertools.product(range(-radius, radius + 1), repeat=3):
            first = next((v for v in offset if v), 0)
            if first > 0:
                pairs.append(offset)
        pairs.sort(key=lambda p: (sum(v * v for v in p), p))
        if len(pairs) >= count // 2:
            return [q for p in pairs[:count // 2] for q in (p, tuple(-v for v in p))]
        radius += 1


def periodic_batch(model, args, torch):
    """Build complete periodic edge pairs without external structure/data files."""
    from ase.data import atomic_numbers

    side, per_cell = args.side, args.side ** 3
    cells = math.ceil(args.edges / (per_cell * args.neighbors))
    if getattr(model, "shift_head", None) is not None and per_cell > 512:
        raise ValueError("X1 QEq supports at most 512 atoms per cell; use --side <= 8")
    grid = torch.tensor(list(itertools.product(range(side), repeat=3)), dtype=torch.long)
    offsets = torch.tensor(neighbor_offsets(args.neighbors), dtype=torch.long)
    cutoff = float(model.model_options["embedding"]["r_max"])
    if float(offsets.float().norm(dim=1).max()) * args.spacing >= cutoff:
        raise ValueError("Some synthetic neighbors exceed r_max; decrease --spacing or --neighbors")
    unwrapped = grid[:, None, :] + offsets[None, :, :]
    wrapped = unwrapped.remainder(side)
    dst = wrapped[..., 0] * side * side + wrapped[..., 1] * side + wrapped[..., 2]
    src = torch.arange(per_cell).repeat_interleave(args.neighbors)
    edge = torch.stack((src, dst.reshape(-1)))
    shift = torch.div(unwrapped, side, rounding_mode="floor").reshape(-1, 3)
    edge_index = torch.cat([edge + g * per_cell for g in range(cells)], dim=1)
    positions = grid.float().repeat(cells, 1) * args.spacing
    symbols = list(model.idp.chemical_symbol_to_type)
    species = torch.arange(len(positions)) % len(symbols)
    numbers = torch.tensor([atomic_numbers[s] for s in symbols])[species]
    width = int(model.idp.reduced_matrix_element)
    rng = torch.Generator().manual_seed(args.seed + 1)
    batch = {
        "pos": positions,
        "cell": (torch.eye(3) * (side * args.spacing)).repeat(cells, 1, 1),
        "pbc": torch.ones(cells, 3, dtype=torch.bool),
        "edge_index": edge_index,
        "edge_cell_shift": shift.repeat(cells, 1).float(),
        "atomic_numbers": numbers[:, None],
        "batch": torch.arange(cells).repeat_interleave(per_cell),
        "ptr": torch.arange(cells + 1) * per_cell,
        "flow_time": torch.zeros(cells, 1),
        "node_h0": torch.randn(len(positions), width, generator=rng) * 0.1,
        "edge_h0": torch.randn(edge_index.shape[1], width, generator=rng) * 0.1,
    }
    batch = {key: value.to(args.device) for key, value in batch.items()}
    model.idp(batch)
    node_mask = model.idp.mask_to_nrme.to(args.device)[batch["atom_types"].flatten()]
    edge_mask = model.idp.mask_to_erme.to(args.device)[batch["edge_type"].flatten()]
    batch["node_h0"] *= node_mask
    batch["edge_h0"] *= edge_mask
    # Synthetic AO overlap only supplies the shape contract of the response head.
    # It is never used as a physical reference or as a Hamiltonian label.
    batch["phys_node_overlap"] = node_mask.float()
    batch["phys_edge_overlap"] = edge_mask.float() * 0.02
    shape = dict(cells=cells, atoms=len(positions), directed_edges=edge_index.shape[1],
                 neighbors=args.neighbors, prior_width=width, atoms_per_cell=per_cell)
    return batch, shape


def select_backend(model, backend):
    """Change execution policy only; parameters and nonlinear routing stay intact."""
    fused = backend == "cuda"
    os.environ["SO2_CUDA_BACKEND"] = "auto" if fused else "off"
    for module in model.modules():
        if hasattr(module, "so2_fusion_mode"):
            module.so2_fusion_mode = "streamed_m_major_fused_p0" if fused else "staged"
        if hasattr(module, "mole_linear_mode"):
            module.mole_linear_mode = "cublas_grouped" if fused else "split_loop"
        if hasattr(module, "m_linear_mode"):
            module.m_linear_mode = "standard"


def timing_summary(samples):
    """Inclusive quartiles; a one-iteration smoke test has a degenerate interval."""
    if any(not math.isfinite(value) or value < 0 for value in samples):
        raise RuntimeError("Timing samples must be finite and nonnegative")
    q1, _, q3 = (statistics.quantiles(samples, n=4, method="inclusive")
                 if len(samples) > 1 else [samples[0]] * 3)
    return {"median": statistics.median(samples), "q1": q1, "q3": q3}


def measure(model, data, backend, args, torch, counters):
    select_backend(model, backend)
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    model.train()
    before = dict(counters)

    def reset():
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        return {key: value.clone() for key, value in data.items()}

    def elapsed(call):
        if args.device.startswith("cuda"):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            result = call()
            end.record()
            end.synchronize()
            return result, start.elapsed_time(end)
        start = time.perf_counter()
        result = call()
        return result, (time.perf_counter() - start) * 1000

    forward, backward = [], []
    outputs = None
    for iteration in range(args.warmup + args.iterations):
        # Release the previous output dictionary before allocating the next graph.
        # DeePTB stores intermediate tensors in that dictionary as well as outputs.
        if outputs is not None:
            del outputs, loss, batch
        batch = reset()
        outputs, fwd_ms = elapsed(lambda: model(batch))
        loss = outputs["node_features"].square().mean() + outputs["edge_features"].square().mean()
        _, bwd_ms = elapsed(loss.backward)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite {backend} loss")
        if iteration >= args.warmup:
            forward.append(fwd_ms)
            backward.append(bwd_ms)
    result = {
        "forward_ms": statistics.median(forward),
        "backward_ms": statistics.median(backward),
        "forward_backward_ms": statistics.median([f + b for f, b in zip(forward, backward)]),
        "forward_samples_ms": forward,
        "backward_samples_ms": backward,
        "loss": float(loss.detach()),
        "timing_statistics_ms": {
            "forward": timing_summary(forward),
            "backward": timing_summary(backward),
            "forward_backward": timing_summary([f + b for f, b in zip(forward, backward)]),
        },
    }
    snapshot = {
        "training_outputs": {key: outputs[key].detach().cpu() for key in ("node_features", "edge_features")},
        "parameter_gradients": {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None},
    }
    for name, grad in snapshot["parameter_gradients"].items():
        if not bool(torch.isfinite(grad).all()):
            raise RuntimeError(f"Non-finite {backend} gradient: {name}")
    del outputs, loss, batch
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    model.eval()
    with torch.no_grad():
        inference = model({key: value.clone() for key, value in data.items()})
    snapshot["inference_outputs"] = {key: inference[key].detach().cpu() for key in ("node_features", "edge_features")}
    del inference
    for group in ("training_outputs", "inference_outputs"):
        if any(not bool(torch.isfinite(value).all()) for value in snapshot[group].values()):
            raise RuntimeError(f"Non-finite {backend} {group}")
    result["dispatch"] = {key: value - before.get(key, 0) for key, value in counters.items() if value > before.get(key, 0)}
    if backend == "cuda":
        expected = "activation_forward" if model.embedding.edge_router_prior_activate else "dense_pairs"
        if result["dispatch"].get(expected, 0) == 0:
            raise RuntimeError(f"No successful {expected} dispatch; the accelerated route fell back")
    elif any(result["dispatch"].get(key, 0) for key in
             ("dense_pairs", "activation_forward", "grouped_gemm", "grouped_gemm_multi")):
        raise RuntimeError("Reference execution unexpectedly called accelerated tensor-product/GEMM entry points")
    if args.device.startswith("cuda"):
        result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
        result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2 ** 30
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    return result, snapshot


def differences(reference, accelerated, torch):
    result = {}
    for group in reference:
        if reference[group].keys() != accelerated[group].keys():
            raise RuntimeError(f"Different {group} tensor keys between backends")
        metrics = {}
        for key, ref in reference[group].items():
            value = accelerated[group][key]
            if ref.shape != value.shape or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"Invalid accelerated tensor {group}/{key}")
            delta = (value.double() - ref.double()).abs()
            metrics[key] = {"max_abs": float(delta.max()) if delta.numel() else 0.0,
                            "relative_l2": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(ref.double()).clamp_min(1e-30))}
        worst = max(metrics, key=lambda key: metrics[key]["max_abs"], default=None)
        result[group] = {"max_abs": metrics[worst]["max_abs"] if worst else 0.0,
                         "max_relative_l2": max((v["relative_l2"] for v in metrics.values()), default=0.0),
                         "worst_tensor": worst, "tensors": metrics}
    return result


def main():
    args = arguments()
    # Set strict FP32 before CUDA initialization, including child JIT compilation.
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["SO2_CUDA_FAST_TF32"] = "0"
    os.environ["DPTB_CUBLAS_GROUPED_FAST_TF32"] = "0"
    os.environ["DPTB_SO2_FUSION_MODE"] = "staged"
    os.environ["SO2_CUDA_PROFILE"] = "0"
    os.environ["DPTB_SO2_PROFILE"] = "0"
    os.environ["DPTB_SO2_ACTIVATION_FUSED_P0"] = "1"
    os.environ["DPTB_SO2_ACTIVATION_FUSED_P0_GEMM"] = "per_slot"
    os.environ["SO2_CUDA_BACKEND"] = "off"
    import torch

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    use_cuda = torch.device(args.device).type == "cuda"
    if use_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu --backend reference for a smoke test")
        torch.cuda.set_device(args.device)
        properties = torch.cuda.get_device_properties(args.device)
        torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_memory_gib * 2 ** 30 / properties.total_memory))
        gpu = properties.name
    else:
        gpu = None
        if args.backend != "reference":
            raise RuntimeError("The CUDA benchmark needs a CUDA device; CPU supports --backend reference")
    forward_mode = "indexed_sandwich_multi" if args.forward_mode == "auto" else args.forward_mode
    os.environ["SO2_CUDA_FORWARD_MODE"] = forward_mode
    os.environ["DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE"] = forward_mode
    counters = install_counters() if args.backend != "reference" else {}
    from dptb.nn.build import build_model
    from dptb.nn.embedding.unitb_options import unitb_options
    import dptb

    report = {
        "created": datetime.now(timezone.utc).isoformat(),
        "precision": "float32; TF32 disabled", "torch_version": torch.__version__,
        "deeptb_version": getattr(dptb, "__version__", None),
        "so2cuda_version": getattr(sys.modules.get("so2_cuda_ops"), "__version__", None),
        "gpu": gpu, "shared_gpu": args.shared_gpu, "seed": args.seed,
        "warmup_iterations": args.warmup, "measured_iterations": args.iterations,
        "dense_forward_mode": forward_mode, "cuda_allocator_limit_gib": args.max_memory_gib,
        "timing_scope": "Model forward and loss backward; setup, input cloning and buffer reset excluded; no optimizer update",
        "data": "Synthetic periodic geometry and random H0; overlap is synthetic; not a physical accuracy test",
        "quantile_method": "statistics.quantiles(n=4, method='inclusive')",
        "models": {},
    }
    models = ("dense", "x1") if args.model == "both" else (args.model,)
    for name in models:
        config_path = Path(__file__).parent / "configs" / (name + ".json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        common = copy.deepcopy(config["common_options"])
        common["device"] = args.device
        options = copy.deepcopy(config["model_options"])
        # Construct a reference-compatible model even when CUDA ops are absent.
        options["embedding"]["so2_fusion_mode"] = "staged"
        options["embedding"]["mole_linear_mode"] = "split_loop"
        torch.manual_seed(args.seed)
        model = build_model(common_options=common, model_options=options, train_options={}, no_check=False)
        data, shape = periodic_batch(model, args, torch)
        effective_embedding = unitb_options(config["model_options"]["embedding"])
        evidence = {"shape": shape, "irreps_hidden": effective_embedding["irreps_hidden"],
                    "embedding_options": effective_embedding,
                    "embedding_method": config["model_options"]["embedding"]["method"],
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "parameters": sum(p.numel() for p in model.parameters()), "backends": {}}
        reference = accelerated = None
        backends = ("reference", "cuda") if args.backend == "both" else (args.backend,)
        for backend in backends:
            if use_cuda:
                torch.cuda.reset_peak_memory_stats()
            result, snapshot = measure(model, data, backend, args, torch, counters)
            evidence["backends"][backend] = result
            if backend == "reference":
                reference = snapshot
            else:
                accelerated = snapshot
            print(f"{name}/{backend}: forward={result['forward_ms']:.3f} ms, "
                  f"backward={result['backward_ms']:.3f} ms, total={result['forward_backward_ms']:.3f} ms, "
                  f"edges={shape['directed_edges']}, dispatch={result['dispatch']}", flush=True)
        if reference is not None and accelerated is not None:
            evidence["differences"] = differences(reference, accelerated, torch)
            evidence["speedup"] = evidence["backends"]["reference"]["forward_backward_ms"] / evidence["backends"]["cuda"]["forward_backward_ms"]
            print(f"{name}: speedup={evidence['speedup']:.3f}x, "
                  f"max output difference={evidence['differences']['training_outputs']['max_abs']:.6g}, "
                  f"max gradient difference={evidence['differences']['parameter_gradients']['max_abs']:.6g}", flush=True)
        report["models"][name] = evidence
        del model, data, reference, accelerated, snapshot
        if use_cuda:
            torch.cuda.empty_cache()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print("CUDA memory limit exceeded. Reduce --edges (or --side/--neighbors for a tiny smoke test).", file=sys.stderr)
        raise
