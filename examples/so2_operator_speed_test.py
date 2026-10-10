#!/usr/bin/env python3
"""SO(2) operator equivalence and exclusive-GPU FP32 benchmarks."""
from __future__ import annotations

import argparse
from copy import copy
from datetime import datetime, timezone
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
import operator_eqv3

from operator_baselines import (ActivationOperator, NaiveOperator, SO2CUDAOperator, CueqOperator, ExplicitGEMMOperator,
                                canonical_weights, prepare_geometry, uniform_irreps)
from operator_eqv3 import EQV3_COMMIT


NONUNIFORM_CASES = (
    ("decreasing_l2", "Decreasing channels, lmax=2", "128x0e+64x1o+32x2e", None),
    ("decreasing_l4", "Decreasing channels, lmax=4", "128x0e+64x1o+32x2e+16x3o+16x4e", None),
    ("v_hidden", "V-shaped hidden channels", "128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e", None),
    ("v_output", "V-shaped input to nonuniform output",
     "128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e",
     "151x0e+37x1o+41x2e+29x3o+13x4e+5x5o+1x6e"),
)


def benchmark_cases(suite="all"):
    """Requested public shape groups; identical rows share one measurement."""
    aliases = {"all": ("A1", "A2", "A3", "A4"), "uniform": ("A1", "A2", "A3"),
               "nonuniform": ("A4",)}
    groups = aliases.get(suite, tuple(suite.split(",")))
    if not groups or any(group not in ("A1", "A2", "A3", "A4") for group in groups):
        raise ValueError("--suite must be all, uniform, nonuniform, or comma-separated A1..A4")
    cases = {}

    def add(group, case_id, label, ii, io, mmax, edges):
        ii, io = str(o3.Irreps(ii)), str(o3.Irreps(io or ii))
        key = ii, io, mmax, edges
        if key not in cases:
            cases[key] = {"id": case_id, "groups": [], "label": label,
                          "config": {"irreps_in": ii, "irreps_out": io, "mmax": mmax, "edges": edges,
                                     "lmax": max(o3.Irreps(ii).lmax, o3.Irreps(io).lmax),
                                     "channels": o3.Irreps(ii)[0].mul if group != "A4" else None}}
        cases[key]["groups"].append(group)

    if "A1" in groups:
        for lmax, channels in itertools.product((2, 4, 6), (32, 64, 128)):
            ii = uniform_irreps(lmax, channels)
            add("A1", f"uniform_l{lmax}_c{channels}_e50000", f"lmax={lmax}, C={channels}",
                ii, ii, lmax, 50000)
    if "A2" in groups:
        for channels, edges in itertools.product((32, 128), (20000, 50000, 130000)):
            ii = uniform_irreps(6, channels)
            add("A2", f"uniform_l6_c{channels}_e{edges}", f"lmax=6, C={channels}", ii, ii, 6, edges)
    if "A3" in groups:
        ii = uniform_irreps(6, 128)
        add("A3", "uniform_l6_m2_c128_e50000", "lmax=6, mmax=2, C=128", ii, ii, 2, 50000)
    if "A4" in groups:
        for (name, label, ii, io), edges in itertools.product(NONUNIFORM_CASES, (20000, 50000, 130000)):
            add("A4", f"{name}_e{edges}", label, ii, io, min(o3.Irreps(ii).lmax, o3.Irreps(io or ii).lmax), edges)
    return list(cases.values())


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
                  so2cuda_candidate="true_dense_pairs"):
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
                     descriptor_name="escn_tp", so2cuda_candidate="true_dense_pairs"):
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
        except operator_eqv3.UnsupportedConfiguration as exc:
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
    report["coverage"] = {"requested": list(implementations), "compared": list(values),
                          "reference": "Independent naive float64 arithmetic and float64 geometry",
                          "edges": edges, "seed": 42}
    report["passed"] = (all(v["passed"] for v in report["comparisons"].values()) and
                        all(v.get("equivariance", {}).get("passed", True) for v in report["implementations"].values()))
    return report


class GPUExclusivityError(RuntimeError):
    pass


def gpu_snapshot(label, allow_shared=False):
    prop = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(prop.uuid)
    if not uuid.startswith("GPU-"):
        uuid = "GPU-" + uuid
    result = subprocess.run(["nvidia-smi", "-q", "-x", "-i", uuid],
                            check=True, capture_output=True, text=True, timeout=30)
    gpu = ET.fromstring(result.stdout).find("gpu")
    processes = [{"pid": row.findtext("pid"), "type": row.findtext("type"),
                  "memory": row.findtext("used_memory")} for row in gpu.findall("processes/process_info")]
    exclusive = bool(processes) and all(row["pid"] == str(os.getpid()) for row in processes)
    if not exclusive and not allow_shared:
        raise GPUExclusivityError(f"GPU is not exclusive at {label}: {processes}")
    proof = {"label": label, "uuid": uuid, "processes": processes, "exclusive": exclusive}
    if allow_shared:
        proof["allow_shared_gpu"] = True
        if not exclusive:
            proof["reason"] = "Shared GPU permitted for a bounded functional smoke test; timings are not formal benchmark evidence"
    return proof


def measure(op, x, upstream, warmup, iterations, allow_shared=False):
    if allow_shared and len(x) > 512:
        raise ValueError("Shared-GPU functional smoke tests require at most 512 edges")
    snapshot = lambda label: gpu_snapshot(label, allow_shared=True) if allow_shared else gpu_snapshot(label)
    record = {"exclusive_gpu_proof": [snapshot("before")],
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
    record["exclusive_gpu_proof"].append(snapshot("after"))
    return record


def compile_counters():
    """Counters identify executed compiled graphs, beyond a request flag."""
    from torch._dynamo.utils import counters
    return {"unique_graphs": int(counters["stats"]["unique_graphs"]),
            "calls_captured": int(counters["stats"]["calls_captured"]),
            "frames_ok": int(counters["frames"]["ok"]),
            "aot_autograd_ok": int(counters["aot_autograd"]["ok"])}


def compile_delta(before, after):
    return {key: after[key] - before[key] for key in before}


def prepare_measurement(name, ii, io, mmax, host_data, eqv3_root, method="naive",
                        rotation="pytorch", descriptor_name="escn_tp", so2cuda_candidate="true_dense_pairs"):
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
    source_provenance = source.get("provenance")
    if "cases" in source:
        matches = [row for row in source["cases"] if
                   o3.Irreps(row["config"]["irreps_in"]) == ii and
                   o3.Irreps(row["config"]["irreps_out"]) == io and row["config"]["mmax"] == mmax]
        if not matches:
            return {}
        source = next((row for row in matches if row.get("cueq_alternatives")), matches[0])
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
                "source_provenance": source_provenance,
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
    scripts = ("so2_operator_speed_test.py", "operator_baselines.py", "operator_cueq.py",
               "operator_eqv3.py", "operator_layout.py")
    result = {"so2cuda_sha": revision, "dirty": dirty.stdout.strip(), "so2cuda_file": so2_cuda_ops.__file__,
              "torch": torch.__version__, "e3nn": e3nn.__version__,
              "benchmark_sha256": {name: hashlib.sha256((root / "examples" / name).read_bytes()).hexdigest()
                                   for name in scripts}}
    try:
        import cuequivariance as cue
        result["cuequivariance"] = cue.__version__
        result["cuequivariance_file"] = cue.__file__
    except ImportError:
        result["cuequivariance"] = None
    return result


def run(args):
    allow_shared = getattr(args, "allow_shared_gpu", False)
    if allow_shared and args.edges > 512:
        raise ValueError("--allow-shared-gpu requires at most 512 edges")
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
    if (args.include_compile or (args.impl == "all" and not args.check_only)) and "eqv3" in impls:
        impls.insert(impls.index("eqv3") + 1, "eqv3+compile")
    report = {"schema": "so2-operator-benchmark-v1", "status": "running", "provenance": source_identity(),
              "timing_contract": "native-layout-v2",
              "config": {"irreps_in": str(ii), "irreps_out": str(io), "lmax": max(ii.lmax, io.lmax),
                         "channels": ii[0].mul if len({mul for mul, _ in ii}) == 1 else None,
                         "mmax": mmax, "edges": args.edges},
              "gpu": torch.cuda.get_device_name(), "precision": "strict FP32; TF32 disabled",
              "geometry": "Precomputed per implementation; input and all weight gradients included, no geometry gradient",
              "warmup": args.warmup, "iterations": args.iterations, "implementations": {}}
    if allow_shared:
        report["measurement_class"] = "bounded functional smoke; shared GPU allowed; not formal timing evidence"
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
        if name == "eqv3+compile" and checks["implementations"].get("eqv3", {}).get("status") == "N/A":
            report["implementations"][name] = dict(checks["implementations"]["eqv3"])
            continue
        if name in checks["implementations"] and checks["implementations"][name]["status"] == "N/A":
            report["implementations"][name] = checks["implementations"][name]
            continue
        if name == "cueq":
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
                        row.update(measure(op, x, upstream, args.warmup, args.iterations, allow_shared=allow_shared))
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
                report["implementations"][name] = {"status": ("oom" if any(r["status"] == "oom" for r in alternatives)
                                                    else "failed" if any(r["status"] == "failed_equivalence" for r in alternatives)
                                                    else "N/A"),
                                                    "reason": "No correct available candidate completed; see cueq_alternatives"}
        else:
            op = x = upstream = None
            try:
                start = time.perf_counter()
                counters_before = compile_counters() if name.endswith("+compile") else None
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
                    row["compile_execution"] = {"counter_delta_before_timing":
                        compile_delta(counters_before, compile_counters())}
                    if row["compile_execution"]["counter_delta_before_timing"]["unique_graphs"] < 1:
                        raise RuntimeError("EquiformerV3 requested compilation but executed no compiled graph")
                row.update(measure(op, x, upstream, args.warmup, args.iterations, allow_shared=allow_shared))
                if name.endswith("+compile"):
                    row["compile_execution"]["counter_delta_after_timing"] = compile_delta(counters_before, compile_counters())
                    row["compile_execution"]["executed"] = True
                row["status"] = "passed"
            except torch.OutOfMemoryError as exc:
                row = {"status": "oom", "reason": str(exc)}
            except operator_eqv3.UnsupportedConfiguration as exc:
                row = {"status": "N/A", "reason": str(exc)}
            finally:
                del op, x, upstream
                gc.collect()
                torch.cuda.empty_cache()
            report["implementations"][name] = row
            print(f"{name}: {row['status']}", flush=True)
        if args.json:
            atomic_json(args.json, report)
    so2 = report["implementations"].get("so2cuda", {})
    if so2.get("status") == "passed":
        for row in report["implementations"].values():
            if row.get("status") == "passed":
                row["speedup_vs_so2cuda"] = {mode: row[mode]["median_ms"] / so2[mode]["median_ms"]
                                               for mode in ("forward", "forward_backward")
                                               if mode in row and mode in so2}
    report["status"] = "failed" if any(r["status"] == "failed" for r in report["implementations"].values()) else "completed"
    if args.json:
        atomic_json(args.json, report)
    if report["status"] == "failed":
        raise RuntimeError("An installed implementation has no correct executable candidate")
    return report


def run_suite(args):
    """Each invocation is one GPU task, with incremental atomic receipts."""
    cases = benchmark_cases(args.suite)
    task_id = datetime.now(timezone.utc).isoformat() + f"/pid-{os.getpid()}"
    report = {"schema": "so2-operator-suite-v2", "status": "running", "suite": args.suite,
              "task_id": task_id, "provenance": source_identity(),
              "gpu": torch.cuda.get_device_name(), "precision": "strict FP32; TF32 disabled",
              "warmup": args.warmup, "iterations": args.iterations, "cases": [],
              "timing_contract": "native-layout-v2",
              "geometry": "Precomputed in each implementation's native layout, outside timing",
              "table_sessions": {group: task_id for case in cases for group in case["groups"]}}
    if getattr(args, "allow_shared_gpu", False):
        report["measurement_class"] = "bounded functional smoke; shared GPU allowed; not formal timing evidence"
    report["provenance"]["equiformerv3_sha"] = EQV3_COMMIT
    report["provenance"]["default_so2cuda_api"] = "so2_cuda_ops.deeptb.true_dense_pairs"
    checked = {}
    if args.json:
        atomic_json(args.json, report)
    for case in cases:
        case_args = copy(args)
        case_args.irreps_in = case["config"]["irreps_in"]
        case_args.irreps_out = case["config"]["irreps_out"]
        case_args.mmax = case["config"]["mmax"]
        case_args.json = None
        case_args.edges = args.check_edges if args.check_only else (args.suite_edges or case["config"]["edges"])
        key = case_args.irreps_in, case_args.irreps_out, case_args.mmax
        if args.check_only and key in checked:
            row = dict(case, task_id=task_id, equivalent_case_id=checked[key])
        else:
            print(f"CASE {case['id']} edges={case_args.edges}", flush=True)
            try:
                result = run(case_args)
            except Exception as exc:
                report["cases"].append(dict(case, task_id=task_id, status="failed",
                                             reason=f"{type(exc).__name__}: {exc}"))
                report["status"] = "failed"
                if args.json:
                    atomic_json(args.json, report)
                raise
            row = dict(case, task_id=task_id, status=result["status"], equivalence=result["equivalence"])
            if args.check_only:
                checked[key] = case["id"]
                row["check_edges"] = case_args.edges
                row["implementations"] = result["equivalence"]["implementations"]
            else:
                row["config"] = result["config"]
                for field in ("implementations", "cueq_alternatives"):
                    if field in result:
                        row[field] = result[field]
        report["cases"].append(row)
        if args.json:
            atomic_json(args.json, report)
        gc.collect()
        torch.cuda.empty_cache()
    report["status"] = "passed" if args.check_only else "completed"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["check_edges"] = args.check_edges if args.check_only else None
    if args.json:
        atomic_json(args.json, report)
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
    parser.add_argument("--suite", help="all, uniform (A1+A2+A3), nonuniform (A4), or comma-separated A1..A4")
    parser.add_argument("--list-cases", action="store_true", help="Print suite definitions without requiring CUDA")
    parser.add_argument("--suite-edges", type=int, help="Override all suite edge counts for a smoke test")
    parser.add_argument("--check-edges", type=int, default=512, help="Small edge count for --suite --check-only")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Permit sharing only for functional smoke tests with <=512 edges; record actual sharing")
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
    if args.allow_shared_gpu:
        bounded_edges = (args.check_edges if args.suite and args.check_only else
                         args.suite_edges if args.suite else args.edges)
        if bounded_edges is None or bounded_edges > 512:
            parser.error("--allow-shared-gpu requires --edges <=512 or an explicit --suite-edges <=512")
    if args.cueq_choice is not None:
        choice = tuple(args.cueq_choice.split(","))
        if (len(choice) != 3 or choice[0] not in ("escn_tp", "escn_tp_compact")
                or choice[1] not in ("naive", "uniform_1d", "fused_tp", "indexed_linear")
                or choice[2] not in ("pytorch", "cueq")):
            parser.error("--cueq-choice must be descriptor,method,rotation")
    if args.suite:
        try:
            cases = benchmark_cases(args.suite)
        except ValueError as exc:
            parser.error(str(exc))
        if args.check_edges <= 0 or (args.suite_edges is not None and args.suite_edges <= 0):
            parser.error("Suite/check edge counts must be positive")
        if args.list_cases:
            print(json.dumps(cases, indent=2))
            return
        run_suite(args)
    else:
        if args.list_cases:
            parser.error("--list-cases requires --suite")
        run(args)


if __name__ == "__main__":
    main()
