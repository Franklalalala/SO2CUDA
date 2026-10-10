#!/usr/bin/env python3
"""SO(2) operator equivalence and exclusive-GPU FP32 benchmarks."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import torch
from e3nn import o3

from operator_baselines import (ActivationOperator, NaiveOperator, SO2CUDAOperator, CueqOperator, ExplicitGEMMOperator,
                                canonical_weights, prepare_geometry, uniform_irreps,
                                SO2CUDA_CANDIDATES, so2cuda_candidate_environment)
from operator_eqv3 import UnsupportedConfiguration


def atomic_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def snapshot(op, x, upstream):
    op.zero_grad(set_to_none=True)
    x = op.input_to_native(x.detach()).detach().requires_grad_(True)
    out = op.forward_native(x)
    out.backward(op.output_to_native(upstream.to(out.dtype)))
    grads = op.canonical_gradients()
    if any(g is None for g in grads) or x.grad is None:
        raise AssertionError("Input and every canonical weight must receive gradients")
    result = {"output": op.output_from_native(out.detach()).cpu(),
              "input_gradient": op.input_from_native(x.grad.detach()).cpu(),
              "weight_gradients": tuple(g.detach().cpu() for g in grads)}
    op.zero_grad(set_to_none=True)
    return result


def metric(a, b):
    if a.shape != b.shape:
        raise AssertionError(f"Shape mismatch: {a.shape} != {b.shape}")
    a, b = a.double(), b.double()
    delta = a - b
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    absolute = float(delta.abs().max()) if a.numel() else 0.
    relative = float(delta.norm() / b.norm().clamp_min(1e-30))
    # Near-zero tensors use an absolute gate. Nonzero tensors use relative L2;
    # arbitrary individual components can cancel to zero in a valid FP32 result.
    passed = finite and (relative <= 5e-5 or absolute <= 2e-5)
    return {"max_abs": absolute, "relative_l2": relative, "finite": finite, "passed": passed}


def compare(a, b):
    if len(a["weight_gradients"]) != len(b["weight_gradients"]):
        raise AssertionError("Canonical weight-gradient coverage differs")
    result = {key: metric(a[key], b[key]) for key in ("output", "input_gradient")}
    result["weight_gradients"] = [metric(x, y) for x, y in zip(a["weight_gradients"], b["weight_gradients"])]
    result["passed"] = (all(result[key]["passed"] for key in ("output", "input_gradient"))
                         and all(row["passed"] for row in result["weight_gradients"]))
    return result


def make_operator(name, irreps_in, irreps_out, mmax, weights, geometry, eqv3_root=None,
                  method="naive", rotation="pytorch", descriptor_name="escn_tp",
                  so2cuda_candidate="dense_pairs"):
    args = irreps_in, irreps_out, mmax, weights, geometry
    if name == "naive":
        return NaiveOperator(*args)
    if name == "so2cuda":
        return SO2CUDAOperator(*args, candidate=so2cuda_candidate)
    if name == "activation":
        return ActivationOperator(*args)
    if name == "explicit_gemm":
        return ExplicitGEMMOperator(*args)
    if name == "cueq":
        return CueqOperator(*args, method=method, rotation=rotation, descriptor_name=descriptor_name)
    if name in ("eqv3", "eqv3+compile"):
        from operator_eqv3 import Eqv3Operator
        return Eqv3Operator(irreps_in, irreps_out, mmax, weights, geometry.rotation_matrix,
                            eqv3_root, compile_model=name.endswith("+compile"))
    raise ValueError(name)


def equivalence_case(irreps_in, irreps_out, mmax, edges=512, eqv3_root=None,
                     implementations=("naive", "so2cuda", "eqv3", "cueq"),
                     method="naive", rotation="pytorch", float64=True, equivariance=True,
                     descriptor_name="escn_tp", so2cuda_candidate="dense_pairs"):
    """One deterministic case, including canonical gradient and rotation checks."""
    torch.manual_seed(42)
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    lmax = max(irreps_in.lmax, irreps_out.lmax)
    weights = canonical_weights(irreps_in, irreps_out, mmax, device="cuda")
    x = torch.randn(edges, irreps_in.dim, device="cuda")
    vec = torch.randn(edges, 3, device="cuda")
    upstream = torch.randn(edges, irreps_out.dim, device="cuda") / irreps_out.dim ** .5
    geometry = prepare_geometry(vec, lmax)
    values, report = {}, {"implementations": {}, "comparisons": {}}
    if float64:
        ref = NaiveOperator(irreps_in, irreps_out, mmax, tuple(w.double() for w in weights),
                            prepare_geometry(vec.double(), lmax))
        values["float64"] = snapshot(ref, x.double(), upstream.double())
        del ref
    for name in implementations:
        if name.startswith("eqv3") and not eqv3_root:
            report["implementations"][name] = {"status": "N/A", "reason": "Provide --eqv3-root"}
            continue
        if name == "cueq" and importlib.util.find_spec("cuequivariance_torch") is None:
            report["implementations"][name] = {"status": "N/A", "reason": "Optional cuEquivariance is not installed"}
            continue
        try:
            op = make_operator(name, irreps_in, irreps_out, mmax, weights, geometry,
                               eqv3_root, method, rotation, descriptor_name, so2cuda_candidate)
        except UnsupportedConfiguration as exc:
            report["implementations"][name] = {"status": "N/A", "reason": str(exc)}
            continue
        values[name] = snapshot(op, x, upstream)
        report["implementations"][name] = {"status": "passed", "metadata": getattr(op, "metadata", {})}
        del op
        if equivariance:
            q = o3.angles_to_matrix(*[torch.tensor(a, dtype=torch.float64, device="cuda") for a in (.37, 1.11, -.51)])
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64)
                dx = irreps_in.D_from_matrix(q.cpu()).to(device=x.device, dtype=torch.float32)
                dy = irreps_out.D_from_matrix(q.cpu()).to(device=x.device, dtype=torch.float32)
            finally:
                torch.set_default_dtype(previous_dtype)
            n = min(16, edges)
            first = make_operator(name, irreps_in, irreps_out, mmax, weights,
                                  prepare_geometry(vec[:n], lmax), eqv3_root, method, rotation, descriptor_name,
                                  so2cuda_candidate)
            second = make_operator(name, irreps_in, irreps_out, mmax, weights,
                                   prepare_geometry(vec[:n] @ q.float().T, lmax), eqv3_root, method, rotation, descriptor_name,
                                   so2cuda_candidate)
            with torch.no_grad():
                check = metric(second(x[:n] @ dx.T), first(x[:n]) @ dy.T)
            report["implementations"][name]["equivariance"] = check
            del first, second
    for a, b in itertools.combinations(values, 2):
        report["comparisons"][a + "__" + b] = compare(values[a], values[b])
    report["passed"] = (all(v["passed"] for v in report["comparisons"].values()) and
                        all(v.get("equivariance", {}).get("passed", True) for v in report["implementations"].values()))
    return report


class GPUExclusivityError(RuntimeError):
    pass


def gpu_snapshot(label):
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(prop.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", uuid],
                            check=True, capture_output=True, text=True, timeout=30)
    gpu = ET.fromstring(result.stdout).find("gpu")
    processes = [{"pid": row.findtext("pid"), "type": row.findtext("type"),
                  "memory": row.findtext("used_memory")} for row in gpu.findall("processes/process_info")]
    if not processes or any(row["pid"] != str(os.getpid()) for row in processes):
        raise GPUExclusivityError(f"GPU is not exclusive at {label}: {processes}")
    return {"label": label, "uuid": uuid, "processes": processes, "exclusive": True}


def measure(op, x, upstream, warmup, iterations):
    record = {"exclusive_gpu_proof": [gpu_snapshot("before")],
              "feature_layout": op.metadata.get("feature_layout", "e3nn mul_ir"),
              "memory_scope": {
                  "includes": ["selected implementation parameters and layout indices",
                               "selected implementation geometry", "native input and upstream gradient",
                               "output, parameter/input gradients, saved activations and workspaces"],
                  "canonical_data": "CPU only; layout conversion temporaries freed before measurement",
                  "other_implementations": "no live parameters, inputs or geometry",
                  "allocated_baseline_bytes": torch.cuda.memory_allocated(),
                  "reserved_note": "allocator cache cleared before each implementation; reserved bytes include allocator reuse"}}
    x = x.detach().requires_grad_(True)
    for backward, label in ((False, "forward"), (True, "forward_backward")):
        samples = []
        peak_allocated = peak_reserved = 0
        for i in range(warmup + iterations):
            op.zero_grad(set_to_none=True)
            x.grad = None
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.set_grad_enabled(backward):
                out = op.forward_native(x)
                if backward:
                    out.backward(upstream)
            end.record()
            end.synchronize()
            if i >= warmup:
                samples.append(begin.elapsed_time(end))
                peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated())
                peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved())
            del out
        q = torch.tensor(samples, dtype=torch.float64).quantile(torch.tensor([.25, .5, .75], dtype=torch.float64)).tolist()
        record[label] = {"median_ms": q[1], "q1_ms": q[0], "q3_ms": q[2], "samples_ms": samples,
                         "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}
    op.zero_grad(set_to_none=True)
    record["exclusive_gpu_proof"].append(gpu_snapshot("after"))
    return record


def prepare_measurement(name, ii, io, mmax, host_data, eqv3_root, method="naive",
                        rotation="pytorch", descriptor_name="escn_tp", so2cuda_candidate="dense_pairs"):
    """Transfer one candidate, convert its inputs, then release setup tensors."""
    weights_cpu, x_cpu, upstream_cpu, vectors_cpu = host_data
    weights = tuple(w.cuda() for w in weights_cpu)
    geometry = prepare_geometry(vectors_cpu.cuda(), max(ii.lmax, io.lmax))
    op = make_operator(name, ii, io, mmax, weights, geometry, eqv3_root,
                       method, rotation, descriptor_name, so2cuda_candidate)
    with torch.no_grad():
        x = op.input_to_native(x_cpu.cuda()).detach()
        upstream = op.output_to_native(upstream_cpu.cuda()).detach()
    del weights, geometry
    gc.collect()
    torch.cuda.empty_cache()
    return op, x, upstream


def previous_rejections(path, ii, io, mmax):
    if path is None:
        return {}
    source = json.loads(Path(path).read_text())
    config = source["config"]
    if (o3.Irreps(config["irreps_in"]) != ii or o3.Irreps(config["irreps_out"]) != io
            or config["mmax"] != mmax):
        raise ValueError("Rejection evidence must match input/output irreps and mmax")
    rows = {}
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for row in source.get("cueq_alternatives", []):
        if row["method"] in ("uniform_1d", "indexed_linear") and row["status"] in ("unavailable", "failed_equivalence"):
            # Geometry/layout changes cannot make an unsupported descriptor
            # method or an incorrect local linear executor become valid.
            inherited = dict(row)
            inherited["reused_rejection"] = {"evidence_sha256": digest,
                "source_provenance": source.get("provenance"),
                "reason": "unchanged descriptor/method rejection; no timing reused"}
            rows[row["descriptor"], row["method"], row["rotation"]] = inherited
    return rows


def source_identity():
    import so2_cuda_ops
    import e3nn
    root = Path(__file__).resolve().parents[1]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    revision = sha.stdout.strip()
    if not revision and (root / ".benchmark_revision").exists():
        revision = (root / ".benchmark_revision").read_text().strip()
    return {"so2cuda_sha": revision, "dirty": dirty.stdout.strip(), "so2cuda_file": so2_cuda_ops.__file__,
            "torch": torch.__version__, "e3nn": e3nn.__version__}


def run(args):
    for name in ("NVIDIA_TF32_OVERRIDE", "SO2_CUDA_FAST_TF32", "DPTB_CUBLAS_GROUPED_FAST_TF32"):
        os.environ[name] = "0"
    for name in ("DPTB_SO2_FUSION_MODE", "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", "SO2_CUDA_FORWARD_MODE"):
        os.environ.pop(name, None)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; imports and CPU reference remain available")
    torch.manual_seed(42)
    ii = o3.Irreps(args.irreps_in) if args.irreps_in else uniform_irreps(args.lmax, args.channels)
    io = o3.Irreps(args.irreps_out) if args.irreps_out else ii
    mmax = min(ii.lmax, io.lmax) if args.mmax is None else args.mmax
    if not 0 <= mmax <= min(ii.lmax, io.lmax):
        raise ValueError("mmax must be between zero and min(input lmax, output lmax)")
    impls = ["naive", "so2cuda", "eqv3", "cueq"] if args.impl == "all" else args.impl.split(",")
    if args.include_compile and "eqv3" in impls:
        impls.insert(impls.index("eqv3") + 1, "eqv3+compile")
    report = {"schema": "so2-operator-benchmark-v1", "status": "running", "provenance": source_identity(),
              "timing_contract": "native-layout-v2",
              "config": {"irreps_in": str(ii), "irreps_out": str(io), "mmax": mmax, "edges": args.edges},
              "gpu": torch.cuda.get_device_name(), "precision": "strict FP32; TF32 disabled",
              "geometry": "Precomputed per implementation; input and all weight gradients included, no geometry gradient",
              "warmup": args.warmup, "iterations": args.iterations, "implementations": {}}
    # All full-shape configurations first pass an independent small equivalence case.
    checks = equivalence_case(ii, io, mmax, args.edges if args.check_only else min(args.edges, 128),
                              args.eqv3_root, [v for v in impls if v != "eqv3+compile"])
    report["equivalence"] = checks
    if not checks["passed"]:
        report["status"] = "failed_equivalence"
        if args.json:
            atomic_json(args.json, report)
        raise AssertionError("Operator equivalence failed; see JSON")
    if args.check_only:
        if "so2cuda" in impls:
            report["so2cuda_alternatives"] = []
            for candidate in SO2CUDA_CANDIDATES:
                with so2cuda_candidate_environment(candidate):
                    validation = equivalence_case(ii, io, mmax, args.edges, args.eqv3_root,
                        implementations=("naive", "so2cuda"), so2cuda_candidate=candidate)
                report["so2cuda_alternatives"].append({"candidate": candidate,
                    "equivalence": validation, "status": "passed" if validation["passed"] else "failed_equivalence"})
            if any(row["status"] != "passed" for row in report["so2cuda_alternatives"]):
                report["status"] = "failed_equivalence"
                if args.json:
                    atomic_json(args.json, report)
                raise AssertionError("SO2CUDA candidate equivalence failed")
        report["status"] = "passed"
        if args.json:
            atomic_json(args.json, report)
        print(json.dumps({"status": report["status"], "implementations": checks["implementations"]}, default=str))
        return report
    torch.manual_seed(42)
    host_data = (canonical_weights(ii, io, mmax, device="cpu"),
                 torch.randn(args.edges, ii.dim),
                 torch.randn(args.edges, io.dim) / io.dim ** .5,
                 torch.randn(args.edges, 3))
    rejected = previous_rejections(getattr(args, "cueq_rejections_json", None), ii, io, mmax)
    gc.collect()
    torch.cuda.empty_cache()
    for name in impls:
        if name in checks["implementations"] and checks["implementations"][name]["status"] == "N/A":
            report["implementations"][name] = checks["implementations"][name]
            continue
        if name == "so2cuda":
            alternatives = []
            for candidate in SO2CUDA_CANDIDATES:
                row = {"candidate": candidate}
                op = x = upstream = None
                try:
                    with so2cuda_candidate_environment(candidate):
                        validation = equivalence_case(ii, io, mmax, 128, args.eqv3_root,
                            implementations=("naive", "so2cuda"), so2cuda_candidate=candidate)
                        row["equivalence"] = validation
                        if not validation["passed"]:
                            row["status"] = "failed_equivalence"
                        else:
                            op, x, upstream = prepare_measurement(name, ii, io, mmax, host_data,
                                args.eqv3_root, so2cuda_candidate=candidate)
                            row["metadata"] = op.metadata
                            row["environment"] = {"SO2_CUDA_FORWARD_MODE": os.environ["SO2_CUDA_FORWARD_MODE"]}
                            row.update(measure(op, x, upstream, args.warmup, args.iterations))
                            row["status"] = "passed"
                except torch.OutOfMemoryError as exc:
                    row.update(status="oom", reason=f"{type(exc).__name__}: {exc}")
                except GPUExclusivityError:
                    raise
                except Exception as exc:
                    row.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    del op, x, upstream
                    gc.collect()
                    torch.cuda.empty_cache()
                alternatives.append(row)
                report["so2cuda_alternatives"] = alternatives
                print(f"so2cuda {candidate}: {row['status']}", flush=True)
                if args.json:
                    atomic_json(args.json, report)
            valid = [r for r in alternatives if r["status"] == "passed"]
            if valid:
                report["implementations"][name] = dict(min(valid,
                    key=lambda r: r["forward_backward"]["median_ms"]))
                report["implementations"][name]["selection"] = "Minimum correct forward+backward median across public pair APIs"
            else:
                report["implementations"][name] = {"status": "oom" if any(r["status"] == "oom" for r in alternatives) else "failed",
                    "reason": "No correct SO2CUDA candidate completed; see so2cuda_alternatives"}
        elif name == "cueq":
            alternatives = []
            combinations = (list(itertools.product(
                    ("escn_tp", "escn_tp_compact"),
                    ("naive", "uniform_1d", "fused_tp", "indexed_linear"), ("pytorch", "cueq")))
                    if getattr(args, "cueq_choice", None) is None else [tuple(args.cueq_choice.split(","))])
            for descriptor_name, method, rotation in combinations:
                row = {"descriptor": descriptor_name, "method": method, "rotation": rotation}
                key = descriptor_name, method, rotation
                if key in rejected:
                    alternatives.append(rejected[key])
                    report["cueq_alternatives"] = alternatives
                    print(f"cueq {descriptor_name}/{method}/{rotation}: reused rejection", flush=True)
                    if args.json:
                        atomic_json(args.json, report)
                    continue
                op = x = upstream = None
                try:
                    validation = equivalence_case(ii, io, mmax, 64, args.eqv3_root,
                        implementations=("naive", "cueq"), method=method, rotation=rotation,
                        descriptor_name=descriptor_name)
                    row["equivalence"] = validation
                    if not validation["passed"]:
                        row["status"] = "failed_equivalence"
                    else:
                        op, x, upstream = prepare_measurement(name, ii, io, mmax, host_data,
                            args.eqv3_root, method, rotation, descriptor_name)
                        row["metadata"] = op.metadata
                        row.update(measure(op, x, upstream, args.warmup, args.iterations))
                        row["status"] = "passed"
                except torch.OutOfMemoryError as exc:
                    row.update(status="oom", reason=str(exc))
                except GPUExclusivityError:
                    raise
                except Exception as exc:
                    row.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
                finally:
                    del op, x, upstream
                    gc.collect()
                    torch.cuda.empty_cache()
                alternatives.append(row)
                print(f"cueq {descriptor_name}/{method}/{rotation}: {row['status']}", flush=True)
                report["cueq_alternatives"] = alternatives
                if args.json:
                    atomic_json(args.json, report)
            valid = [r for r in alternatives if r["status"] == "passed"]
            if valid:
                selected = min(valid, key=lambda r: r["forward_backward"]["median_ms"])
                report["implementations"][name] = dict(selected)
                report["implementations"][name]["selection"] = (
                    "Minimum correct forward+backward median across descriptor/method/rotation candidates"
                    if getattr(args, "cueq_choice", None) is None else
                    "Fixed descriptor/method/rotation given by --cueq-choice (selected by an earlier full scan)")
            else:
                report["implementations"][name] = {"status": "oom" if any(r["status"] == "oom" for r in alternatives) else "failed",
                                                    "reason": "No correct available candidate completed; see cueq_alternatives"}
        else:
            op = x = upstream = None
            try:
                start = time.perf_counter()
                op, x, upstream = prepare_measurement(name, ii, io, mmax, host_data, args.eqv3_root)
                row = {"metadata": op.metadata}
                if name.endswith("+compile"):
                    # Trigger forward/backward compilation before both timed loops.
                    row["construction_seconds"] = time.perf_counter() - start
                    start = time.perf_counter()
                    compiled_check = snapshot(op, op.input_from_native(x), op.output_from_native(upstream))
                    row["compile_and_first_training_call_seconds"] = time.perf_counter() - start
                    eager, eager_x, eager_upstream = prepare_measurement("eqv3", ii, io, mmax, host_data, args.eqv3_root)
                    row["equivalence_vs_eager"] = compare(compiled_check,
                        snapshot(eager, eager.input_from_native(eager_x), eager.output_from_native(eager_upstream)))
                    del eager, eager_x, eager_upstream, compiled_check
                    if not row["equivalence_vs_eager"]["passed"]:
                        raise RuntimeError("Compiled EquiformerV3 failed equivalence")
                    start = time.perf_counter()
                    with torch.no_grad():
                        op.forward_native(x)
                    torch.cuda.synchronize()
                    op.zero_grad(set_to_none=True)
                    row["compile_and_first_inference_call_seconds"] = time.perf_counter() - start
                    gc.collect()
                    torch.cuda.empty_cache()
                row.update(measure(op, x, upstream, args.warmup, args.iterations))
                row["status"] = "passed"
            except torch.OutOfMemoryError as exc:
                row = {"status": "oom", "reason": str(exc)}
            except UnsupportedConfiguration as exc:
                row = {"status": "N/A", "reason": str(exc)}
            finally:
                del op, x, upstream
                gc.collect()
                torch.cuda.empty_cache()
            report["implementations"][name] = row
            print(f"{name}: {row['status']}", flush=True)
        if args.json:
            atomic_json(args.json, report)
    report["status"] = "failed" if any(r["status"] == "failed" for r in report["implementations"].values()) else "completed"
    if args.json:
        atomic_json(args.json, report)
    if report["status"] == "failed":
        raise RuntimeError("An installed implementation has no correct executable candidate")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", default="all")
    parser.add_argument("--lmax", type=int, default=2)
    parser.add_argument("--mmax", type=int)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--irreps-in")
    parser.add_argument("--irreps-out")
    parser.add_argument("--edges", type=int, default=20000)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--eqv3-root", default=os.environ.get("SO2CUDA_EQV3_ROOT"))
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--cueq-rejections-json", type=Path,
                        help="Reuse matching unsupported/incorrect method evidence; never reuse timings")
    parser.add_argument("--cueq-choice", help="descriptor,method,rotation: time only this cuEquivariance "
                        "combination instead of scanning all of them")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.edges <= 0 or args.channels <= 0 or args.lmax < 0:
        parser.error("Require positive edges/channels and nonnegative lmax")
    if not args.check_only and (args.warmup < 5 or args.iterations < 20):
        parser.error("Timing requires at least 5 warmups and 20 iterations")
    if args.cueq_choice is not None:
        choice = tuple(args.cueq_choice.split(","))
        if (len(choice) != 3 or choice[0] not in ("escn_tp", "escn_tp_compact")
                or choice[1] not in ("naive", "uniform_1d", "fused_tp", "indexed_linear")
                or choice[2] not in ("pytorch", "cueq")):
            parser.error("--cueq-choice must be descriptor,method,rotation")
    run(args)


if __name__ == "__main__":
    main()
