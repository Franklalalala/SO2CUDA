#!/usr/bin/env python3
"""Compare SO(2) backends on an immutable stream of real training batches.

The production MultiTrainer constructs the model, loss, calibrated dynamic
loader and optimizers. Its unmodified iteration performs every measured step.
Only the ordinary execution adapters are swapped. No dataset is bundled here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import gc
import hashlib
import importlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import time

# Configure math policy before the first CUDA initialization.
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["SO2_CUDA_FAST_TF32"] = "0"
os.environ["DPTB_CUBLAS_GROUPED_FAST_TF32"] = "0"

import torch

from deeptb_speed_test import DispatchAudit, select_backend, timing_summary


LABELS = {"dense": "UniTB-dense", "unitb": "UniTB", "slem": "UniTB-SLEM"}
BACKENDS = {"so2cuda": "cuda", "naive": "reference", "cueq": "cueq"}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=LABELS)
    parser.add_argument("--head", required=True, choices=("onsite", "hopping"))
    parser.add_argument("--backends", nargs="+", choices=BACKENDS,
                        default=list(BACKENDS))
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--equivalence-structures", type=int, default=1)
    parser.add_argument("--equivalence-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Allow shorter measurement windows")
    parser.add_argument("--metadata-cache", type=Path, action="append", default=[])
    parser.add_argument("--validation-root", type=Path,
                        help="Relocate an unavailable validation copy to the same corpus")
    parser.add_argument("--train-overlap-sidecar-root", type=Path,
                        help="Relocate the training overlap sidecar to the same corpus")
    parser.add_argument("--validation-overlap-sidecar-root", type=Path,
                        help="Relocate the validation overlap sidecar to the same corpus")
    parser.add_argument("--cueq-method", default="auto")
    parser.add_argument("--cueq-descriptor", choices=("escn_tp", "escn_tp_compact"), default="escn_tp")
    parser.add_argument("--dptb-sha", required=True)
    parser.add_argument("--so2cuda-sha", required=True)
    parser.add_argument("--shared-gpu", action="store_true", help="For small equivalence only")
    args = parser.parse_args()
    if min(args.warmup, args.iterations, args.equivalence_structures) < 1:
        parser.error("Warmup, iterations and equivalence structures must be positive")
    if not args.smoke and not args.equivalence_only and (args.warmup < 4 or args.iterations < 12):
        parser.error("Formal measurements require at least 4 warmup and 12 measured steps")
    if args.shared_gpu and not args.equivalence_only:
        parser.error("Shared GPUs are allowed only for equivalence checks")
    if len(set(args.backends)) != len(args.backends):
        parser.error("Backend names must be unique")
    return args


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def production_config(source, args):
    """Relocate dataset copies while retaining every scientific option."""
    config = copy.deepcopy(source)
    if config["train_options"]["batch_size"] != 32:
        raise ValueError("The production config must use batch_size=32")
    changes = {}
    relocations = (("validation_root", "validation", "root"),
                   ("train_overlap_sidecar_root", "train", "overlap_sidecar_root"),
                   ("validation_overlap_sidecar_root", "validation", "overlap_sidecar_root"))
    for argument, split, field in relocations:
        destination = getattr(args, argument, None)
        if destination is not None:
            options = config["data_options"][split]
            old, new = options.get(field), str(destination)
            options[field] = new
            if old != new:
                changes[f"data_options.{split}.{field}"] = [old, new]
    return config, changes


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def batch_record(batch):
    """Fingerprint tensor contents as well as structure order and graph counts."""
    digest = hashlib.sha256()
    for name in sorted(batch.keys):
        value = batch[name]
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    indices = getattr(batch, "__dptb_sample_indices__", None)
    digest.update(json.dumps(indices).encode())
    return {"structures": int(batch.num_graphs), "atoms": int(batch.num_nodes),
            "edges": int(batch.edge_index.shape[1]), "sample_indices": indices,
            "sha256": digest.hexdigest()}


def stream_record(batches, warmup):
    records = [batch_record(batch) for batch in batches]
    measured = records[warmup:]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "all_batches": records, "measured_batches": measured,
            "mean_structures": sum(row["structures"] for row in measured) / len(measured),
            "mean_edges": sum(row["edges"] for row in measured) / len(measured),
            "same_stream_for_all_backends": True}


def small_batch(batch, count):
    from dptb.utils.torch_geometric import Batch
    data = batch.to_data_list()[:count]
    result = Batch.from_data_list(data).contiguous()
    indices = getattr(batch, "__dptb_sample_indices__", None)
    result.__dptb_sample_indices__ = None if indices is None else indices[:len(data)]
    return result


class MetadataCache:
    """Reuse immutable costs keyed by the original dataset signature, not config."""
    def __init__(self, paths):
        self.values = {}
        self.sources = []
        for path in paths:
            self.values.update(json.loads(path.read_text()))
            self.sources.append({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.hits = self.misses = 0
        self.signatures = {}

    @contextmanager
    def install(self):
        loader = importlib.import_module("dptb.data.dataloader")
        original = loader._metadata_cost_parts

        def cost(dataset, idx, estimator, data=None, **kwargs):
            if data is not None or not hasattr(dataset, "_lmdb_path_map"):
                return original(dataset, idx, estimator, data=data, **kwargs)
            ident = id(dataset)
            if ident not in self.signatures:
                signature = {"paths": dataset._lmdb_path_map, "indices": dataset.index_map,
                             "selection": list(dataset.indices()),
                             "version": dataset.dynamic_batch_cost_version}
                self.signatures[ident] = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
            key = f"{self.signatures[ident]}:{estimator.mode}:{idx}"
            if key in self.values:
                self.hits += 1
                value, parts = self.values[key]
                return value, dict(parts)
            value, parts = original(dataset, idx, estimator, data=data, **kwargs)
            self.values[key] = [value, dict(parts)]
            self.misses += 1
            return value, parts

        loader._metadata_cost_parts = cost
        try:
            yield
        finally:
            loader._metadata_cost_parts = original

    def record(self):
        return {"hits": self.hits, "misses": self.misses, "entries": len(self.values),
                "sources": self.sources, "signature_scheme": "original production dataset signature"}


def initialize(config, out, cache):
    """Stop immediately after the production trainer is fully constructed."""
    from dptb.entrypoints.multi_train import multi_train
    from dptb.nnops.multi_trainer import MultiTrainer

    class Ready(Exception):
        pass

    trainers = []
    original = MultiTrainer.__init__

    def create(self, *args, **kwargs):
        original(self, *args, **kwargs)
        trainers.append(self)
        raise Ready()

    MultiTrainer.__init__ = create
    try:
        with cache.install():
            try:
                multi_train(INPUT=str(config), init_model=None, restart=None,
                            output=str(out / "run"), log_level=20, log_path=str(out / "train.log"))
            except Ready:
                pass
    finally:
        MultiTrainer.__init__ = original
    if len(trainers) != 1:
        raise RuntimeError("Expected one single-process training instance")
    trainer = trainers[0]
    if trainer.distributed_expert or trainer.use_reference:
        raise ValueError("This benchmark supports single-process production training without a reference loader")
    for optimizer in trainer.optimizers:
        if type(optimizer).__name__ != "HybridMuon":
            raise ValueError("Production optimizer must be HybridMuon")
        optimizer.execution_mode = "fast"
    return trainer


class InitialState:
    """Restore identical parameters, buffers, optimizer, schedule and noise RNG."""
    def __init__(self, trainer):
        from dptb.nnops.training_state import capture_rng_state
        self.model = cpu_clone(trainer.model.state_dict())
        self.optimizers = [cpu_clone(optimizer.state_dict()) for optimizer in trainer.optimizers]
        self.schedulers = [cpu_clone(scheduler.state_dict()) for scheduler in trainer.lr_schedulers]
        self.rng = capture_rng_state()
        self.iteration = int(trainer.iter)
        self.batch_cursor = getattr(trainer, "_batch_in_epoch", 0)
        # HybridMuon publishes these numerical controls as Python attributes;
        # they are deliberately absent from the model checkpoint state.
        self.router_progress = {
            name: {field: copy.deepcopy(getattr(module, field))
                   for field in ("opt_step", "bias_lr_scale")}
            for name, module in trainer.model.named_modules()
            if hasattr(module, "opt_step") and hasattr(module, "bias_lr_scale")}

    def restore(self, trainer):
        from dptb.nnops.training_state import restore_rng_state
        trainer.model.zero_grad(set_to_none=True)
        trainer.model.load_state_dict(self.model, strict=True)
        for optimizer, state in zip(trainer.optimizers, self.optimizers):
            optimizer.load_state_dict(copy.deepcopy(state))
            optimizer.execution_mode = "fast"
        for scheduler, state in zip(trainer.lr_schedulers, self.schedulers):
            scheduler.load_state_dict(copy.deepcopy(state))
        trainer.iter = self.iteration
        trainer._batch_in_epoch = self.batch_cursor
        trainer._t_last_iter_end = None
        modules = dict(trainer.model.named_modules())
        for name, progress in self.router_progress.items():
            for field, value in progress.items():
                setattr(modules[name], field, copy.deepcopy(value))
        restore_rng_state(self.rng)


def metric(reference, value, *, atol=2e-6, rtol=2e-4):
    a, b = reference.detach().double(), value.detach().double()
    difference = (a - b).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"max_abs": float(difference.max()) if difference.numel() else 0.0,
            "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-30)),
            "finite": finite, "passed": finite and bool((difference <= atol + rtol * a.abs()).all()),
            "atol": atol, "rtol": rtol}


def compare(first, second):
    gradients_a, gradients_b = first["gradients"], second["gradients"]
    if gradients_a.keys() != gradients_b.keys():
        raise RuntimeError("Backend changed the named parameter contract")
    result = {"loss": metric(torch.tensor(first["loss"]), torch.tensor(second["loss"]), rtol=1e-4)}
    rows = {}
    for name, value in gradients_a.items():
        other = gradients_b[name]
        if value is None or other is None:
            rows[name] = {"passed": value is None and other is None, "unused": value is None and other is None}
        else:
            rows[name] = metric(value, other)
    result["parameter_gradients"] = {"passed": all(row["passed"] for row in rows.values()),
                                      "parameters": len(rows),
                                      "max_abs": max((row.get("max_abs", 0) for row in rows.values()), default=0),
                                      "max_relative_l2": max((row.get("relative_l2", 0) for row in rows.values()), default=0),
                                      "finite": all(row.get("finite", True) for row in rows.values()),
                                      "failed_parameters": [name for name, row in rows.items() if not row["passed"]],
                                      "per_parameter": rows}
    result["passed"] = result["loss"]["passed"] and result["parameter_gradients"]["passed"]
    return result


def probe(trainer, batch, backend):
    """Audit an untimed first step and capture every raw pre-clipping gradient."""
    from dptb.nn.so2_backend import STATS
    audit = DispatchAudit()
    STATS.reset()
    gradients = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip(parameters, *args, **kwargs):
        if not gradients:
            gradients.update({name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                              for name, parameter in trainer.model.named_parameters()})
        return original_clip(parameters, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = clip
    try:
        with audit.observe(), forward_backward_timer(trainer) as intervals:
            loss = trainer.iteration(batch.clone())
        torch.cuda.synchronize()
    finally:
        torch.nn.utils.clip_grad_norm_ = original_clip
    if loss is None or not bool(torch.isfinite(loss).all()) or not gradients:
        raise RuntimeError("First step did not produce a finite loss and gradients")
    if len(intervals) != len(trainer.optimizers):
        raise RuntimeError("First-step timer did not bracket every production expert backward")
    if backend != "so2cuda" and audit.calls:
        raise RuntimeError("Ordinary execution called SO2CUDA: " + str(dict(audit.calls)))
    successful_so2 = sum(audit.get(name, 0) for name in ("activation_forward", "dense_pairs", "true_dense_forward"))
    if backend == "so2cuda" and successful_so2 == 0:
        raise RuntimeError("No successful SO2CUDA dispatch was observed")
    evidence = {"loss": float(loss.detach()), "dispatch": dict(audit),
                "so2cuda_calls": dict(audit.calls), "route_stats": STATS.snapshot(),
                "gradient_stage": "all named parameters before clipping and optimizer update",
                "forward_backward_timer_verified": True,
                "forward_backward_intervals": len(intervals),
                "finite_gradients": all(value is None or bool(torch.isfinite(value).all())
                                         for value in gradients.values())}
    if backend == "cueq":
        from cueq_baseline import cueq_execution_metadata
        evidence["cueq_execution"] = cueq_execution_metadata(trainer.model)
        if not evidence["cueq_execution"] or any(row["calls"] == 0 for row in evidence["cueq_execution"].values()):
            raise RuntimeError("cuEquivariance did not execute every SO2 layer")
    return {"loss": evidence["loss"], "gradients": gradients}, evidence


@contextmanager
def forward_backward_timer(trainer):
    """CUDA intervals from payload construction through its matching backward.

    Batch loading/preparation, gradient clipping, metrics and optimizer update
    are excluded. The complete step uses synchronized host wall time instead.
    Wrappers preserve the production functions and are restored on exit.
    """
    intervals = []
    pending = []
    original_payload = trainer._build_train_payload
    original_backward = torch.Tensor.backward

    def payload(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        result = original_payload(*args, **kwargs)
        pending.append((result["loss"], start))
        return result

    def backward(tensor, *args, **kwargs):
        result = original_backward(tensor, *args, **kwargs)
        match = next(((index, start) for index, (loss, start) in enumerate(pending) if loss is tensor), None)
        if match is not None:
            index, start = match
            pending.pop(index)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            intervals.append((start, end))
        return result

    trainer._build_train_payload = payload
    torch.Tensor.backward = backward
    try:
        yield intervals
        if pending:
            raise RuntimeError("A training loss was not followed by backward")
    finally:
        trainer._build_train_payload = original_payload
        torch.Tensor.backward = original_backward


def measure(trainer, batches, warmup, stream_hash):
    rows = []
    peak_allocated = peak_reserved = 0.0
    with forward_backward_timer(trainer) as intervals:
        for index, stored in enumerate(batches):
            # Copy and release the preceding CPU batch outside the measured step.
            batch = stored.clone()
            intervals.clear()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.perf_counter()
            loss = trainer.iteration(batch)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - begin) * 1000
            if loss is None or not bool(torch.isfinite(loss).all()):
                raise RuntimeError("A timed training step was skipped or nonfinite")
            if len(intervals) != len(trainer.optimizers):
                raise RuntimeError("Missing forward/backward interval for a production expert")
            row = {"step": index + 1, "loss": float(loss.detach()), "step_ms": elapsed,
                   "forward_backward_ms": sum(start.elapsed_time(end) for start, end in intervals),
                   "structures": int(stored.num_graphs), "edges": int(stored.edge_index.shape[1])}
            if index >= warmup:
                rows.append(row)
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated() / 2 ** 30)
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved() / 2 ** 30)
            print("BENCH_STEP " + json.dumps(row), flush=True)
            del batch, loss
    step = timing_summary([row["step_ms"] for row in rows])
    forward_backward = timing_summary([row["forward_backward_ms"] for row in rows])
    return {"status": "ok", "step_ms": step["median"], "forward_backward_ms": forward_backward["median"],
            "timing_statistics_ms": {"step": step, "forward_backward": forward_backward},
            "peak_allocated_gib": peak_allocated, "peak_reserved_gib": peak_reserved,
            "batch_stream_sha256": stream_hash, "steps": rows,
            "optimizer": "HybridMuon", "optimizer_mode": "fast",
            "muon_graph_counts": [len(getattr(optimizer, "_bucket_graphs", {})) for optimizer in trainer.optimizers]}


def processes():
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    text = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid,
                                   "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"], text=True)
    foreign = [line for line in text.splitlines() if line.strip() and int(line.split(",")[0]) != os.getpid()]
    return {"exclusive": not foreign, "processes": text.splitlines()}


def main():
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Real-batch measurements require CUDA")
    torch.set_num_threads(min(int(os.environ.get("OMP_NUM_THREADS", "8")), 8))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.shared_gpu:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 20 * 2 ** 30 / total_memory))
    args.out.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.config.read_text())
    config, changes = production_config(source, args)
    # Keep the scientific config and calibration options intact; no run loop is
    # entered, so max_steps and checkpoint/display frequencies need no override.
    input_path = args.out / "input.json"
    dump(input_path, config)
    before = processes()
    if not args.shared_gpu and not before["exclusive"]:
        raise RuntimeError("Benchmark GPU has another compute process")
    report = {"schema_version": 1, "status": "in_progress", "model": args.model, "label": LABELS[args.model], "head": args.head,
              "backends": {}, "warmup_steps": args.warmup, "measured_steps": args.iterations,
              "equivalence": {"status": "in_progress", "backends": {}, "comparisons": {}},
              "provenance": {"dptb_sha": args.dptb_sha, "so2cuda_sha": args.so2cuda_sha,
                             "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                             "cuda": torch.version.cuda, "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
                             "source_config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                             "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "config_changes": changes, "TF32": False, "optimizer": "HybridMuon",
                             "optimizer_mode": "fast", "gpu_before": before,
                             "execution_scope": "SO(2) tensor products and associated PDQ-MoE expert linears; router, nonlinearities, other layers, loss and optimizer unchanged",
                             "production_dynamic_batch": config["train_options"].get("dynamic_batch"),
                             "timing_scope": {"step": "synchronized MultiTrainer.iteration wall time, including fast HybridMuon",
                                              "forward_backward": "CUDA interval from training payload construction through loss.backward; excludes batch preparation, clipping, metrics and optimizer"}}}
    result_path = args.out / "RESULT.json"
    dump(result_path, report)
    cache = MetadataCache(args.metadata_cache)
    try:
        trainer = initialize(input_path, args.out, cache)
    except Exception as error:
        oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        report["status"] = "failed"
        report["backends"] = {backend: {"status": "oom" if oom else "error", "reason": str(error),
                                        "stage": "production initialization"} for backend in args.backends}
        report["equivalence"].update({"status": "failed", "passed": False, "reason": str(error)})
        dump(result_path, report)
        dump(args.out / "EQUIV_MODEL.json", {"schema_version": 1, "model": args.model, "head": args.head,
                                             "provenance": report["provenance"], **report["equivalence"]})
        raise SystemExit(1) from error
    # Entry-point runtime setup may change matmul precision; enforce strict FP32
    # again after construction, without modifying the production config.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    count = 1 if args.equivalence_only else args.warmup + args.iterations
    with cache.install():
        batches = [batch.cpu() for batch in itertools.islice(trainer.train_loader, count)]
    if len(batches) != count:
        raise RuntimeError("The training loader did not provide the requested batch stream")
    report["metadata_cost_cache"] = cache.record()
    parameter_dtypes = {str(parameter.dtype).removeprefix("torch.") for parameter in trainer.model.parameters()}
    if parameter_dtypes != {"float32"}:
        raise ValueError("All production parameters must use float32")
    report["precision"] = {"dtype": "float32", "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                           "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                           "matmul_precision": torch.get_float32_matmul_precision(),
                           "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                           "SO2_CUDA_FAST_TF32": os.environ.get("SO2_CUDA_FAST_TF32"),
                           "DPTB_CUBLAS_GROUPED_FAST_TF32": os.environ.get("DPTB_CUBLAS_GROUPED_FAST_TF32")}
    report["loader"] = {"batch_sampler": type(trainer.train_loader.batch_sampler).__name__,
                        "dynamic": getattr(trainer.train_loader, "dynamic_batch_options", None)}
    if cache.misses:
        dump(args.out / "metadata_cache_new.json", cache.values)
    report["batch_stream"] = stream_record(batches, 0 if args.equivalence_only else args.warmup)
    equivalence_batch = small_batch(batches[0], args.equivalence_structures)
    report["equivalence"]["batch"] = batch_record(equivalence_batch)
    initial = InitialState(trainer)
    parameters = {name: id(parameter) for name, parameter in trainer.model.named_parameters()}
    snapshots = {}
    for backend in args.backends:
        print("BENCH_BACKEND " + backend, flush=True)
        backend_before = processes()
        try:
            if not args.shared_gpu and not backend_before["exclusive"]:
                raise RuntimeError("Benchmark GPU has another compute process before " + backend)
            initial.restore(trainer)
            select_backend(trainer.model, BACKENDS[backend], cueq_method=args.cueq_method,
                           cueq_descriptor=args.cueq_descriptor)
            if parameters != {name: id(parameter) for name, parameter in trainer.model.named_parameters()}:
                raise RuntimeError("Backend replacement changed parameter objects or names")
            snapshots[backend], evidence = probe(trainer, equivalence_batch, backend)
            report["equivalence"]["backends"][backend] = evidence
            initial.restore(trainer)
            trainer.model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if args.equivalence_only:
                report["backends"][backend] = {"status": "ok", "timed": False}
            else:
                report["backends"][backend] = measure(trainer, batches, args.warmup, report["batch_stream"]["sha256"])
            if backend == "cueq":
                from cueq_baseline import cueq_execution_metadata
                report["backends"][backend]["cueq_execution"] = cueq_execution_metadata(trainer.model)
        except Exception as error:
            oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
            report["backends"][backend] = {"status": "oom" if oom else "error", "reason": str(error),
                                           "batch_stream_sha256": report["batch_stream"]["sha256"]}
            print("BENCH_FAILURE " + json.dumps(report["backends"][backend]), flush=True)
            trainer.model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
        backend_after = processes()
        report["backends"][backend]["gpu_before"] = backend_before
        report["backends"][backend]["gpu_after"] = backend_after
        if not args.shared_gpu and not backend_after["exclusive"]:
            report["backends"][backend]["exclusive_gpu_check_failed"] = True
        dump(result_path, report)
    for first, second in itertools.combinations(snapshots, 2):
        report["equivalence"]["comparisons"][first + "_vs_" + second] = compare(snapshots[first], snapshots[second])
    passed = len(snapshots) == len(args.backends) and all(
        row["passed"] for row in report["equivalence"]["comparisons"].values()) and all(
        row["finite_gradients"] for row in report["equivalence"]["backends"].values())
    report["equivalence"]["status"] = "passed" if passed else "failed"
    report["equivalence"]["passed"] = passed
    report["equivalence"]["initial_state_identical"] = True
    report["provenance"]["gpu_after"] = processes()
    if not args.shared_gpu and not report["provenance"]["gpu_after"]["exclusive"]:
        report["exclusive_gpu_check_failed"] = True
    import dptb
    import so2_cuda_ops
    report["provenance"].update({"deeptb_version": getattr(dptb, "__version__", None),
                                 "so2cuda_version": getattr(so2_cuda_ops, "__version__", None),
                                 "imported_deeptb": str(Path(dptb.__file__).resolve()),
                                 "imported_so2cuda": str(Path(so2_cuda_ops.__file__).resolve())})
    failed = (any(row["status"] == "error" or row.get("exclusive_gpu_check_failed")
                  for row in report["backends"].values()) or not passed or report.get("exclusive_gpu_check_failed"))
    report["status"] = "failed" if failed else "completed"
    dump(result_path, report)
    dump(args.out / "EQUIV_MODEL.json", {"schema_version": 1, "model": args.model, "head": args.head,
                                         "provenance": report["provenance"], **report["equivalence"]})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
