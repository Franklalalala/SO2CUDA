#!/usr/bin/env python3
"""Generate English README performance tables and public evidence from JSON.

Operator input is a so2-operator-suite-v2 report (or an aggregate of suites).
Model input contains the six real-batch model/head reports. An explicit option
also accepts unavailable whole model tables with failure evidence. Private task records,
paths and identities are excluded from the public evidence by an allowlist.
Nothing is timed here; every displayed timing and ratio comes from input JSON.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import statistics


BEGIN = "<!-- SO2CUDA_BENCHMARKS_BEGIN -->"
END = "<!-- SO2CUDA_BENCHMARKS_END -->"
EQV3_SHA = "a7300c58df683dc99cb48027d5bfd4c887486c48"
DEEPTB_SHA = "8d5a0dcda30547f83869c292d48fab5df6eec722"
RELEASE_SHA = "6bc05a7b5135efc153dbb75bf7a20b949979adcf"
UNCHANGED_ROUTE_SHAS = {"179faaaabe269464fd7e0ad1a36926253bd08eb1",
                        "367642e703847d1c6ffc7f84ffdf7e84ad1de713"}
LABELS = {"so2cuda": "SO2CUDA", "naive": "Our pure PyTorch", "cueq": "cuEquivariance",
          "eqv3": "EquiformerV3", "eqv3+compile": "EquiformerV3 + compile"}
OPERATOR_COLUMNS = ("so2cuda", "naive", "cueq", "eqv3", "eqv3+compile")
MODEL_COLUMNS = ("so2cuda", "naive", "cueq")
MODEL_LABELS = {"dense": "UniTB-dense", "unitb": "UniTB", "slem": "UniTB-SLEM"}
MODEL_ALIASES = {"unitb_slem": "slem", "UniTB-SLEM": "slem", "UniTB-dense": "dense", "UniTB": "unitb"}
UNAVAILABLE_REASONS = {
    "runtime_error": "The measurement failed before a complete validated table was available.",
    "progress_timeout": "The measurement failed after exceeding its progress timeout.",
    "task_timeout": "The measurement failed after exceeding its execution time limit.",
    "failed_equivalence": "The measurement failed numerical or backend validation.",
    "not_measured": "No complete validated measurement is available.",
}
GROUP_LABELS = {"A1": "Shape", "A2": "Edge count", "A3": "Truncated m", "A4": "Non-uniform irreps"}
METHODS = {"naive", "uniform_1d", "fused_tp", "indexed_linear"}
DESCRIPTORS = {"escn_tp", "escn_tp_compact"}
CUEQ_CHOICE = {"descriptor": "escn_tp_compact", "method": "naive", "rotation": "pytorch"}
CUEQ_SELECTION_FILE = "docs/benchmarks/CUEQ_SELECTION_SCAN.json"
DECREASING_2 = "128x0e+64x1o+32x2e"
DECREASING_4 = DECREASING_2 + "+16x3o+16x4e"
V_SHAPE = "128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e"
TARGET = "151x0e+37x1o+41x2e+29x3o+13x4e+5x5o+1x6e"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected finite numeric measurement")
    return value


def public_reason(value):
    """Remove file paths, identities and control characters from library diagnostics."""
    value = re.sub(r"(?:/[A-Za-z0-9_.+~-]+){2,}", "<path>", str(value))
    value = re.sub(r"\b[A-Za-z]:[\\/][^\s\"']+", "<path>", value)
    value = re.sub(r"\b[^\s@]+@[^\s@]+\b", "<identity>", value)
    value = re.sub(r"GPU-[A-Za-z0-9-]+", "<gpu>", value)
    value = re.sub(r"\b(?:job|account|hostname|host)\s*[:=]\s*\S+", "<identity>", value, flags=re.I)
    value = " ".join(value.split())
    return value.replace("|", "\\|")[:500]


def status(value):
    if value in ("passed", "ok", "completed", "complete"):
        return "passed"
    if isinstance(value, str) and value.lower() == "oom":
        return "oom"
    if value in ("N/A", "na", "not_applicable", "unavailable", "unsupported"):
        return "N/A"
    return "failed"


def measurement(row, scope, *, required=True, expected_samples=None):
    data = row.get(scope)
    if not isinstance(data, dict):
        legacy = row.get("timing_statistics_ms", {}).get(scope)
        if legacy:
            data = {"median_ms": legacy["median"], "q1_ms": legacy["q1"], "q3_ms": legacy["q3"]}
        elif not required:
            return None
        else:
            raise ValueError("Missing recorded " + scope + " median and quartiles")
    result = {key: number(data[key]) for key in ("median_ms", "q1_ms", "q3_ms")}
    if result["median_ms"] <= 0 or not (0 <= result["q1_ms"] <= result["median_ms"] <= result["q3_ms"]):
        raise ValueError("Invalid timing median or quartiles")
    samples = data.get("samples_ms", row.get(scope + "_samples_ms"))
    if samples is None and row.get("steps"):
        samples = [step[scope + "_ms"] for step in row["steps"]]
    if samples is not None:
        samples = [number(x) for x in samples]
        if not samples:
            raise ValueError("Empty recorded timing samples")
        if any(value < 0 for value in samples):
            raise ValueError("Negative recorded timing sample")
        if expected_samples is not None and len(samples) != expected_samples:
            raise ValueError("Recorded timing sample count disagrees with measured iterations")
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive") if len(samples) > 1 else [samples[0]] * 3
        for key, expected in (("median_ms", statistics.median(samples)), ("q1_ms", q1), ("q3_ms", q3)):
            if not math.isclose(result[key], expected, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Timing summary does not reproduce from recorded samples")
        result["samples_ms"] = samples
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = data.get(key, row.get(key))
        if value is None:
            value = row.get(key.removesuffix("_bytes") + "_gib")
            if value is not None:
                value = number(value) * 2 ** 30
        if value is not None:
            result[key] = number(value)
            if result[key] < 0:
                raise ValueError("Negative recorded peak memory")
    return result


def metadata(row):
    source = {**row.get("metadata", {}), **row}
    result = {}
    for key, allowed in (("descriptor", DESCRIPTORS), ("method", METHODS), ("rotation", {"pytorch", "cueq"})):
        if source.get(key) in allowed:
            result[key] = source[key]
    version = source.get("version")
    if isinstance(version, str) and re.fullmatch(r"\d+(?:\.\d+){1,3}[A-Za-z0-9+.-]*", version):
        result["version"] = version
    if source.get("candidate") in ("true_dense_pairs", "dense_pairs", "dense_pairs_grouped"):
        result["candidate"] = source["candidate"]
    api = source.get("api")
    if api in ("so2_cuda_ops.deeptb.true_dense_pairs", "so2_cuda_ops.deeptb.dense_pairs",
               "so2_cuda_ops.deeptb.activation_forward"):
        result["api"] = api
    if source.get("forward_mode") in (None, "default"):
        result["default_settings"] = True
    return result


def implementation(row, *, model=False, alternative=False, expected_samples=None):
    result = {"status": status(row.get("status")), **metadata(row)}
    if alternative and row.get("status") == "failed_equivalence":
        result["status"] = "failed_equivalence"
    if result["status"] == "failed" and not alternative:
        raise ValueError("A requested implementation failed; resolve it before publishing")
    if result["status"] == "passed":
        if model:
            result["step"] = measurement(row, "step", expected_samples=expected_samples)
        else:
            result["forward"] = measurement(row, "forward", expected_samples=expected_samples)
        result["forward_backward"] = measurement(row, "forward_backward", expected_samples=expected_samples)
        if not alternative:
            for phase in (("step", "forward_backward") if model else ("forward", "forward_backward")):
                if "peak_allocated_bytes" not in result[phase]:
                    raise ValueError("Successful measurement lacks " + phase + " peak allocated memory")
        proof = row.get("exclusive_gpu_proof", [row.get("gpu_before"), row.get("gpu_after")])
        if not isinstance(proof, list) or len(proof) < 2 or any(not isinstance(p, dict) or p.get("exclusive") is not True for p in proof):
            raise ValueError("Successful measurement lacks before/after exclusive GPU proof")
        result["gpu_exclusive_before_and_after"] = True
        if row.get("compile_execution"):
            execution = row["compile_execution"]
            if execution.get("executed") is not True:
                raise ValueError("Requested compilation did not execute")
            result["compile_execution"] = {"executed": True}
            for phase in ("counter_delta_before_timing", "counter_delta_after_timing"):
                result["compile_execution"][phase] = {key: number(execution[phase][key]) for key in
                    ("unique_graphs", "calls_captured", "frames_ok", "aot_autograd_ok")}
            if result["compile_execution"]["counter_delta_before_timing"]["unique_graphs"] < 1:
                raise ValueError("No compiled graph was observed before timing")
    else:
        if result["status"] == "oom":
            result["reason"] = "CUDA out of memory at the requested shape."
        elif result["status"] == "failed_equivalence":
            result["reason"] = "Rejected because numerical equivalence failed."
            result["equivalence"] = equivalence_evidence(row["equivalence"], require_pass=False)
        elif row.get("reason") or row.get("error"):
            result["reason"] = public_reason(row.get("reason", row.get("error")))
        else:
            result["reason"] = "The selected implementation does not support this configuration."
    return result


def provenance(raw):
    output = {}
    aliases = {"so2cuda": "SO2CUDA", "so2cuda_sha": "SO2CUDA", "deeptb": "DeePTB", "dptb_sha": "DeePTB",
               "deeptb_sha": "DeePTB", "equiformerv3": "EquiformerV3", "equiformerv3_sha": "EquiformerV3",
               "eqv3_sha": "EquiformerV3"}
    versions = {}
    for source in (raw.get("payload_source", {}), raw.get("provenance", {}), raw):
        values = {**source.get("commits", {}), **source.get("source_commits", {}), **source}
        for key, value in values.items():
            name = aliases.get(key.lower().replace("_", "") if key.lower().replace("_", "") in aliases else key.lower())
            if name and isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value):
                if name in output and output[name] != value:
                    raise ValueError("Conflicting recorded source commits for " + name)
                output[name] = value
        for key in ("torch", "cuda", "e3nn", "cuequivariance"):
            value = source.get(key, source.get(key + "_version"))
            if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+){1,3}[A-Za-z0-9+.-]*", value):
                versions[key] = value
    return {"source_commits": output, "versions": versions}


def row_provenance(row, raw):
    parent, own = provenance(raw), provenance(row)
    return {key: {**parent[key], **own[key]} for key in ("source_commits", "versions")}


def table_source_commits(rows, raw, *, operator):
    """Keep a commit pin for every complete table, including mixed releases."""
    tables = {}
    recorded = raw.get("table_source_commits", raw.get("table_commits", {}))
    for row in rows:
        commits = row["source_commits"]
        if "SO2CUDA" not in commits:
            raise ValueError("Every table row must pin its SO2CUDA source commit")
        for group in row["groups"] if operator else [row["model"]]:
            if group in tables and tables[group] != commits:
                raise ValueError("One table includes different source commits: " + group)
            tables[group] = commits
            if group in recorded and recorded[group] != commits:
                raise ValueError("Table source commits disagree with its rows: " + group)
    return tables


def common_source_commits(tables):
    values = list(tables.values())
    return {key: value for key, value in values[0].items()
            if all(row.get(key) == value for row in values)} if values else {}


def cueq_selection_basis(raw):
    """Publish the fixed choice and its immutable scan reference, never paths."""
    if raw.get("schema") != "so2-cueq-selection-basis-v1" or raw.get("choice") != CUEQ_CHOICE:
        raise ValueError("The cuEquivariance selection basis must pin compact/naive/PyTorch")
    if (raw.get("config_count") != 12 or raw.get("alternatives_per_config") != 16
            or raw.get("metric") != "forward_backward.median_ms"
            or raw.get("evidence_file") != CUEQ_SELECTION_FILE):
        raise ValueError("cuEquivariance selection must refer to the complete 12-by-16 uniform scan")
    source_hash = raw.get("source_sha256")
    if not isinstance(source_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", source_hash):
        raise ValueError("The cuEquivariance selection scan needs its original SHA256")
    commits = provenance({"source_commits": raw.get("source_commits", {})})["source_commits"]
    if (set(commits) != {"SO2CUDA", "DeePTB", "EquiformerV3"}
            or commits["EquiformerV3"] != EQV3_SHA or commits["DeePTB"] != DEEPTB_SHA):
        raise ValueError("The cuEquivariance selection scan needs its pinned source commits")
    return {"schema": raw["schema"], "choice": dict(CUEQ_CHOICE), "source_sha256": source_hash,
            "source_commits": commits, "evidence_file": CUEQ_SELECTION_FILE,
            "config_count": 12, "alternatives_per_config": 16, "metric": raw["metric"]}


def candidate_key(row):
    return tuple(row.get(key) for key in ("descriptor", "method", "rotation"))


def validate_cueq_candidates(chosen, alternatives, basis=None):
    keys = [candidate_key(row) for row in alternatives]
    expected = (set(itertools.product(DESCRIPTORS, METHODS, ("pytorch", "cueq")))
                if basis is None else {candidate_key(CUEQ_CHOICE)})
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("cuEquivariance candidate records must cover the full scan or the pinned single choice")
    if any(row["status"] not in ("passed", "N/A", "oom", "failed_equivalence") for row in alternatives):
        raise ValueError("Incomplete or failed cuEquivariance candidate attempt")
    if chosen["status"] == "passed":
        if candidate_key(chosen) not in keys:
            raise ValueError("Selected cuEquivariance implementation lacks a candidate record")
        matching = alternatives[keys.index(candidate_key(chosen))]
        if matching["status"] != "passed" or any(chosen[phase] != matching[phase]
                for phase in ("forward", "forward_backward")):
            raise ValueError("Selected cuEquivariance timing disagrees with its candidate record")
        if basis is None:
            successful = [row for row in alternatives if row["status"] == "passed"]
            if chosen["forward_backward"]["median_ms"] != min(row["forward_backward"]["median_ms"] for row in successful):
                raise ValueError("Selected cuEquivariance candidate is not the fastest valid measurement")
    if basis is not None:
        if chosen["status"] != "passed" and candidate_key(chosen) == (None, None, None):
            chosen.update(CUEQ_CHOICE)
        if candidate_key(chosen) != candidate_key(CUEQ_CHOICE):
            raise ValueError("The cuEquivariance column differs from its pinned selection basis")


def hardware(raw):
    value = raw.get("gpu", raw.get("hardware", raw.get("provenance", {}).get("gpu", "")))
    if isinstance(value, dict):
        value = value.get("name", "")
    if "H200" not in str(value):
        raise ValueError("Public performance tables require NVIDIA H200 measurements")
    return "NVIDIA H200"


def require_precision(raw):
    value = raw.get("precision", {})
    if isinstance(value, str):
        if "fp32" in value.lower() and "tf32" in value.lower() and "disabled" in value.lower():
            return
    elif isinstance(value, dict):
        if value.get("dtype") in ("float32", "fp32") and value.get("allow_tf32") is False:
            return
    if raw.get("tf32") is False and str(value).lower() in ("fp32", "float32"):
        return
    raise ValueError("Strict FP32 with TF32 disabled must be recorded")


def irreps(value):
    value = str(value).replace(" ", "")
    if not re.fullmatch(r"\d+x\d+[eo](?:\+\d+x\d+[eo])*", value):
        raise ValueError("Invalid irreps representation")
    return value


def uniform(value):
    entries = [(int(mul), int(level)) for mul, level in re.findall(r"(\d+)x(\d+)[eo]", value)]
    if entries and len({mul for mul, _ in entries}) == 1 and [l for _, l in entries] == list(range(len(entries))):
        return len(entries) - 1, entries[0][0]
    return None


def operator_rows(raw):
    """Flatten aggregate wrappers without publishing any private wrapper fields."""
    for row in raw.get("cases", raw.get("reports", [])):
        if "config" in row and "implementations" in row:
            yield row
        else:
            result = row.get("result", row)
            yield from operator_rows(result)


def task_key(row, raw):
    return row.get("task_id", row.get("session_id", raw.get("task_id", raw.get("session_id"))))


def check_table_sessions(rows, raw, *, operator):
    sessions, devices = {}, {}
    recorded = raw.get("table_sessions", {})
    for row in rows:
        key = task_key(row, raw)
        if key is None:
            raise ValueError("Missing card-task identity for a table row")
        groups = row.get("groups", row.get("grids", [])) if operator else [row["model"]]
        for group in groups:
            if key is not None:
                sessions.setdefault(group, set()).add(str(key))
            impls = row.get("implementations", row.get("backends", {}))
            identities = {row.get("provenance", {}).get("gpu_uuid")}
            for impl in impls.values():
                identities.update(p.get("uuid") for p in impl.get("exclusive_gpu_proof", []) if isinstance(p, dict))
            devices.setdefault(group, set()).update(identity for identity in identities if identity)
    for group, keys in sessions.items():
        if len(keys) != 1:
            raise ValueError("One table includes rows from different card tasks: " + group)
        if group in recorded and str(recorded[group]) not in keys:
            raise ValueError("Table session provenance disagrees with its rows")
        if len(devices[group]) > 1:
            raise ValueError("One table includes different GPUs: " + group)


def operator_configuration(config):
    return tuple(config[key] for key in ("irreps_in", "irreps_out", "mmax", "edges"))


def expected_operator_tables():
    def flat(lmax, channels, mmax, edges):
        value = "+".join(f"{channels}x{level}{'e' if level % 2 == 0 else 'o'}"
                         for level in range(lmax + 1))
        return value, value, mmax, edges
    return {
        "A1": {flat(l, c, l, 50000) for l in (2, 4, 6) for c in (32, 64, 128)},
        "A2": {flat(6, c, 6, n) for c in (32, 128) for n in (20000, 50000, 130000)},
        "A3": {flat(6, 128, 2, 50000)},
        "A4": {(i, o, m, n) for i, o, m in (
            (DECREASING_2, DECREASING_2, 2), (DECREASING_4, DECREASING_4, 4),
            (V_SHAPE, V_SHAPE, 6), (V_SHAPE, TARGET, 6)) for n in (20000, 50000, 130000)},
    }


def operator_evidence(raw, *, required_groups=None):
    """Validate complete tables; a configuration can occur in disjoint tables.

    The default requires the entire published matrix. ``required_groups`` lets
    evidence collectors validate a selected whole table before aggregation.
    Completion here describes tables, never the exit status of a source task.
    """
    required = set(GROUP_LABELS if required_groups is None else required_groups)
    if not required or not required <= set(GROUP_LABELS):
        raise ValueError("Unknown required operator tables")
    if raw.get("status") not in ("completed", "complete", "passed"):
        raise ValueError("Operator measurement suite has not completed")
    hardware(raw)
    require_precision(raw)
    source_rows = list(operator_rows(raw))
    if not source_rows:
        raise ValueError("Operator suite contains no configurations")
    check_table_sessions(source_rows, raw, operator=True)
    cases = []
    for source in source_rows:
        if source.get("status", "completed") not in ("completed", "complete", "passed"):
            raise ValueError("An operator configuration has not completed")
        groups = source.get("groups", source.get("grids", []))
        if not groups or len(groups) != len(set(groups)) or any(group not in required for group in groups):
            raise ValueError("Operator group memberships A1–A4 are required")
        config = source["config"]
        config = {"irreps_in": irreps(config["irreps_in"]), "irreps_out": irreps(config["irreps_out"]),
                  "mmax": number(config["mmax"]), "edges": number(config["edges"])}
        shape = uniform(config["irreps_in"])
        if shape and config["irreps_in"] == config["irreps_out"]:
            config["lmax"], config["channels"] = shape
        warmup = number(source.get("warmup", raw.get("warmup")))
        iterations = number(source.get("iterations", raw.get("iterations")))
        if warmup < 5 or iterations < 20:
            raise ValueError("Operator timings require at least five warmups and twenty measured iterations")
        impls = source["implementations"]
        if set(OPERATOR_COLUMNS) - set(impls):
            raise ValueError("Operator configuration lacks requested implementations")
        row = {"groups": list(groups), "config": config, "warmup": warmup, "iterations": iterations,
               **row_provenance(source, raw),
               "implementations": {name: implementation(impls[name], expected_samples=iterations) for name in OPERATOR_COLUMNS}}
        so2cuda = row["implementations"]["so2cuda"]
        if so2cuda["status"] == "passed" and (so2cuda.get("api") != "so2_cuda_ops.deeptb.true_dense_pairs"
                or not so2cuda.get("default_settings")):
            raise ValueError("The SO2CUDA column must use its public true_dense_pairs API with default settings")
        compiled = row["implementations"]["eqv3+compile"]
        if compiled["status"] == "passed" and compiled.get("compile_execution", {}).get("executed") is not True:
            raise ValueError("Compiled EquiformerV3 timing lacks execution evidence")
        cueq = row["implementations"]["cueq"]
        if cueq["status"] == "passed":
            if not all(key in cueq for key in ("descriptor", "method", "rotation", "version")):
                raise ValueError("cuEquivariance timing lacks its version, descriptor, method or rotation")
            if cueq["version"] != "0.12.0":
                raise ValueError("These tables require cuEquivariance 0.12.0")
        row["cueq_alternatives"] = [implementation(value, alternative=True, expected_samples=iterations)
                                    for value in source.get("cueq_alternatives", [])]
        basis = source.get("cueq_selection_basis")
        if basis is None and len(row["cueq_alternatives"]) != 16:
            basis = raw.get("cueq_selection_basis")
        if basis is not None:
            row["cueq_selection_basis"] = cueq_selection_basis(basis)
        validate_cueq_candidates(cueq, row["cueq_alternatives"], row.get("cueq_selection_basis"))
        if "A4" in groups:
            for name in ("eqv3", "eqv3+compile"):
                if row["implementations"][name]["status"] != "N/A":
                    raise ValueError("EquiformerV3 must be N/A for non-uniform irreps")
                row["implementations"][name]["reason"] = "The original implementation requires equal channels across l on each side; no padding is used."
        cases.append(row)
    keys = [(group, operator_configuration(row["config"])) for row in cases for group in row["groups"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate operator configuration within one table")
    for group, configurations in expected_operator_tables().items():
        if group not in required:
            continue
        actual = {operator_configuration(r["config"]) for r in cases if group in r["groups"]}
        if actual != configurations:
            raise ValueError("Operator " + group + " table is incomplete")
    tables = table_source_commits(cases, raw, operator=True)
    return {"schema": "so2cuda-public-operator-benchmarks-v4", "hardware": "NVIDIA H200", "precision": "FP32",
            "tf32": False, "geometry_precomputed": True, "feature_layout_conversion_timed": False,
            **provenance(raw), "source_commits": common_source_commits(tables),
            "table_source_commits": tables, "cases": cases}


def cueq_selection_scan(raw, basis):
    """Preserve all original candidates without declaring a partial suite complete."""
    summaries = basis.get("cases")
    basis = cueq_selection_basis(basis)
    hardware(raw)
    require_precision(raw)
    sources = list(operator_rows(raw))
    if len(sources) != basis["config_count"]:
        raise ValueError("The selection scan must contain the twelve original complete configurations")
    check_table_sessions(sources, raw, operator=True)
    cases = []
    for source in sources:
        if source.get("status") not in ("completed", "complete", "passed"):
            raise ValueError("The selection scan contains an incomplete configuration")
        config = source["config"]
        config = {"irreps_in": irreps(config["irreps_in"]), "irreps_out": irreps(config["irreps_out"]),
                  "mmax": number(config["mmax"]), "edges": number(config["edges"])}
        shape = uniform(config["irreps_in"])
        if not shape or config["irreps_in"] != config["irreps_out"] or config["mmax"] != shape[0]:
            raise ValueError("The selection scan requires uniform, untruncated configurations")
        config["lmax"], config["channels"] = shape
        warmup = number(source.get("warmup", raw.get("warmup")))
        iterations = number(source.get("iterations", raw.get("iterations")))
        if warmup < 5 or iterations < 20:
            raise ValueError("The selection scan lacks the operator timing protocol")
        chosen_source = source["implementations"]["cueq"]
        chosen = implementation(chosen_source, expected_samples=iterations)
        alternatives = [implementation(row, alternative=True, expected_samples=iterations)
                        for row in source.get("cueq_alternatives", [])]
        validate_cueq_candidates(chosen, alternatives)
        if chosen["status"] != "passed" or candidate_key(chosen) != candidate_key(CUEQ_CHOICE):
            raise ValueError("The original uniform scan does not support the fixed cuEquivariance choice")
        equivalence = equivalence_evidence(chosen_source["equivalence"])
        source_commits = row_provenance(source, raw)["source_commits"]
        if any(value != basis["source_commits"].get(key) for key, value in source_commits.items()):
            raise ValueError("The original selection scan source differs from its selection basis")
        cases.append({"config": config, "warmup": warmup, "iterations": iterations,
                      "selected": chosen, "selected_equivalence": equivalence,
                      "cueq_alternatives": alternatives})
    configurations = [operator_configuration(row["config"]) for row in cases]
    if (len(configurations) != len(set(configurations))
            or not expected_operator_tables()["A1"] <= set(configurations)
            or not set(configurations) <= expected_operator_tables()["A1"] | expected_operator_tables()["A2"]):
        raise ValueError("The selection scan must preserve the original shape and edge-count grid")
    if summaries is not None:
        check_cueq_scan_summaries(summaries, cases)
    return {"schema": "so2cuda-public-cueq-selection-scan-v1", "hardware": "NVIDIA H200",
            "precision": "FP32", "tf32": False, "selection_basis": basis, "cases": cases}


def check_cueq_scan_summaries(summaries, cases):
    expected = {}
    for case in cases:
        successful = sorted(row["forward_backward"]["median_ms"] for row in case["cueq_alternatives"]
                            if row["status"] == "passed")
        expected[operator_configuration(case["config"])] = (
            CUEQ_CHOICE, successful[0], successful[1], len(case["cueq_alternatives"]))
    actual = {}
    for row in summaries:
        key = operator_configuration(row["config"])
        if key in actual:
            raise ValueError("Duplicate cuEquivariance selection summary")
        actual[key] = (row["selected"], number(row["selected_median_ms"]),
                       number(row["runner_up_median_ms"]), row["candidate_count"])
    if actual != expected:
        raise ValueError("cuEquivariance selection summaries differ from the original full scan")


def selection_scan_artifact(raw, path=None):
    references = ([raw["cueq_selection_basis"]] if "cueq_selection_basis" in raw else [])
    references.extend(row["cueq_selection_basis"] for row in operator_rows(raw) if "cueq_selection_basis" in row)
    if not references:
        if path is not None or "cueq_selection_scan" in raw:
            raise ValueError("The selection scan is missing its explicit selection basis")
        return None
    bases = [cueq_selection_basis(value) for value in references]
    if any(value != bases[0] for value in bases):
        raise ValueError("Operator tables refer to different cuEquivariance selection scans")
    embedded = raw.get("cueq_selection_scan")
    if path is not None:
        if digest(path) != bases[0]["source_sha256"]:
            raise ValueError("The original cuEquivariance selection scan SHA256 differs from its basis")
        original = load(path)
        if embedded is not None and embedded != original:
            raise ValueError("Embedded and original cuEquivariance selection scans disagree")
    else:
        if (embedded is None or raw.get("cueq_selection_session", {}).get("raw_sha256")
                != bases[0]["source_sha256"]):
            raise ValueError("The operator aggregate must retain its original cuEquivariance selection scan and digest")
        original = embedded
    scan = cueq_selection_scan(original, references[0])
    for reference in references[1:]:
        if "cases" in reference:
            check_cueq_scan_summaries(reference["cases"], scan["cases"])
    return scan


def model_rows(raw):
    if "model" in raw and "head" in raw and ("backends" in raw or "implementations" in raw):
        yield raw
    else:
        for row in raw.get("cases", raw.get("reports", raw.get("models", []))):
            yield from model_rows(row.get("result", row))


def model_identity(source):
    model = MODEL_ALIASES.get(source["model"], source["model"])
    head = source["head"]
    if model not in MODEL_LABELS or head not in ("onsite", "hopping"):
        raise ValueError("Unknown model or head")
    return model, head


def unavailable_model_case(source, raw):
    model, head = model_identity(source)
    state = source.get("status")
    if state not in ("failed", "unmeasured"):
        raise ValueError("Unavailable model cases must distinguish failed from unmeasured")
    reason_code = source.get("reason_code", "runtime_error" if state == "failed" else "not_measured")
    if reason_code not in UNAVAILABLE_REASONS or ((state == "unmeasured") != (reason_code == "not_measured")):
        raise ValueError("Unavailable model status disagrees with its reason code")
    evidence = source.get("raw_evidence_sha256", {})
    if not isinstance(evidence, dict) or any(
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", key)
            or not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for key, value in evidence.items()):
        raise ValueError("Unavailable model evidence needs safe labels and SHA256 digests")
    recorded = row_provenance(source, raw)
    if state == "failed" and (not evidence or task_key(source, raw) is None
                              or not {"SO2CUDA", "DeePTB"} <= set(recorded["source_commits"])):
        raise ValueError("A failed model table needs its task identity, source commits and raw evidence digests")
    return {"model": model, "head": head, "status": state, "reason_code": reason_code,
            "reason": UNAVAILABLE_REASONS[reason_code], "raw_evidence_sha256": evidence, **recorded}


def model_evidence(raw, *, allow_incomplete=False):
    incomplete = raw.get("status") == "incomplete"
    if raw.get("status") not in ("completed", "complete", "passed") and not (allow_incomplete and incomplete):
        raise ValueError("Model measurement suite has not completed")
    if raw.get("unavailable_cases") and not (allow_incomplete and incomplete):
        raise ValueError("Unavailable model tables require explicit incomplete publication")
    source_rows = list(model_rows(raw))
    unavailable_sources = raw.get("unavailable_cases", []) if incomplete else []
    unavailable = [unavailable_model_case(source, raw) for source in unavailable_sources]
    attempted = source_rows + [source for source in unavailable_sources if task_key(source, raw) is not None]
    check_table_sessions(attempted, raw, operator=False)
    cases = []
    for source in source_rows:
        if source.get("status", "completed") not in ("completed", "complete", "passed"):
            raise ValueError("A model/head measurement has not completed")
        model, head = model_identity(source)
        hardware(source if source.get("gpu") or source.get("provenance", {}).get("gpu") else raw)
        require_precision(source if "precision" in source else raw)
        warmup = number(source.get("warmup_steps", source.get("warmup", raw.get("warmup_steps"))))
        iterations = number(source.get("measured_steps", source.get("iterations", raw.get("measured_steps"))))
        if warmup < 4 or iterations < 12:
            raise ValueError("Real-batch timings require at least four warmups and twelve measured steps")
        stream = source["batch_stream"]
        means = {key: number(stream[key]) for key in ("mean_structures", "mean_edges")}
        if min(means.values()) <= 0 or means["mean_structures"] > 32:
            raise ValueError("Invalid measured dynamic-batch means")
        if "measured_batches" in stream:
            batches = stream["measured_batches"]
            if not isinstance(batches, list) or len(batches) != iterations:
                raise ValueError("Recorded batch count disagrees with measured steps")
            for key, count_key in (("mean_structures", "structures"), ("mean_edges", "edges")):
                counts = [number(batch[count_key]) for batch in batches]
                if any(value <= 0 or (count_key == "structures" and value > 32) for value in counts):
                    raise ValueError("Invalid measured batch counts")
                if not math.isclose(means[key], statistics.mean(counts), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Dynamic-batch mean does not reproduce from recorded batches")
        fingerprint = stream.get("sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("The shared real-batch stream needs its SHA256 fingerprint")
        backends = source.get("backends", source.get("implementations"))
        if set(backends) != set(MODEL_COLUMNS):
            raise ValueError("A model/head row needs all three requested backends")
        row = {"model": model, "head": head, "warmup": warmup, "iterations": iterations,
               "batch_stream": {**means, "sha256": fingerprint}, **row_provenance(source, raw),
               "implementations": {name: implementation(backends[name], model=True, expected_samples=iterations) for name in MODEL_COLUMNS}}
        for backend in backends.values():
            own_hash = backend.get("batch_stream_sha256", fingerprint)
            if own_hash != fingerprint:
                raise ValueError("Model backends used different real-batch streams")
            if status(backend.get("status")) == "passed" and (backend.get("optimizer") != "HybridMuon"
                    or backend.get("optimizer_mode") != "fast"):
                raise ValueError("Every successful backend must use the fast HybridMuon path")
        cases.append(row)
    expected = {(model, head) for model in MODEL_LABELS for head in ("onsite", "hopping")}
    actual = [(r["model"], r["head"]) for r in cases + unavailable]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Real-batch table needs each of three models and two heads exactly once")
    for name in MODEL_LABELS:
        if sum(row["model"] == name for row in cases) not in (0, 2):
            raise ValueError("A model table must validate both heads in the same task or disclose both as unavailable")
    if incomplete and not unavailable:
        raise ValueError("An incomplete model suite must identify unavailable tables")
    pinned = cases + [row for row in unavailable if "SO2CUDA" in row["source_commits"]]
    tables = table_source_commits(pinned, raw, operator=False)
    return {"schema": "so2cuda-public-real-batch-benchmarks-v2", "hardware": "NVIDIA H200", "precision": "FP32",
            "tf32": False, "optimizer": "HybridMuon", "optimizer_mode": "fast", "batch_size_limit": 32,
            **provenance(raw), "source_commits": common_source_commits(tables), "table_source_commits": tables,
            "cases": sorted(cases, key=lambda r: (list(MODEL_LABELS).index(r["model"]), r["head"] == "hopping")),
            **({"status": "incomplete", "unavailable_cases": unavailable} if incomplete else {})}


def model_equivalence_evidence(raw, model):
    if model.get("status") != "incomplete":
        return equivalence_evidence(raw)
    if raw.get("status") != "incomplete":
        raise ValueError("Incomplete model timings need equivalence scoped to the available tables")
    expected = {(row["model"], row["head"]) for row in model["cases"]}
    cases = []
    for source in raw.get("cases", []):
        name, head = model_identity(source)
        cases.append({"model": name, "head": head, **equivalence_evidence(source["equivalence"])})
    actual = [(row["model"], row["head"]) for row in cases]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Model equivalence must cover exactly the validated model/head timings")
    expected_unavailable = {(row["model"], row["head"], row["status"]) for row in model["unavailable_cases"]}
    unavailable = [{"model": model_identity(row)[0], "head": row["head"], "status": row["status"]}
                   for row in raw.get("unavailable_cases", [])]
    actual_unavailable = [(row["model"], row["head"], row["status"]) for row in unavailable]
    if set(actual_unavailable) != expected_unavailable or len(actual_unavailable) != len(expected_unavailable):
        raise ValueError("Unavailable model equivalence must match the unavailable timing tables")
    return {"schema": "so2cuda-public-model-equivalence-v1", "status": "incomplete", "all_passed": False,
            "metric_count": sum(row["metric_count"] for row in cases), "cases": cases,
            "unavailable_cases": unavailable}


def equivalence_evidence(raw, *, require_pass=True):
    """Keep numerical errors and criteria without parameter names or private records."""
    rows, flags, dispatch = [], [], []
    quantities = {"output", "input_gradient", "weight_gradients", "canonical_weight_gradients", "parameter_gradients",
                  "loss", "first_step_loss", "training_outputs", "inference_outputs", "equivariance"}
    def visit(node, quantity="comparison"):
        if isinstance(node, dict):
            if isinstance(node.get("passed"), bool):
                flags.append(node["passed"])
            if "max_abs" in node and ("relative_l2" in node or "max_relative_l2" in node):
                item = {"quantity": quantity, "max_abs": number(node["max_abs"]),
                        "relative_l2": number(node.get("relative_l2", node.get("max_relative_l2")))}
                for key in ("passed", "finite"):
                    if key in node:
                        item[key] = bool(node[key])
                for key in ("rms_error", "reference_max_abs", "reference_rms", "atol", "rtol"):
                    if key in node:
                        item[key] = number(node[key])
                rows.append(item)
            for backend, evidence in node.get("backends", {}).items():
                if backend not in MODEL_COLUMNS or "so2cuda_calls" not in evidence:
                    continue
                calls = sum(number(value) for value in evidence["so2cuda_calls"].values())
                successes = sum(number(value) for value in evidence.get("dispatch", {}).values())
                valid = calls == 0 if backend in ("naive", "cueq") else successes > 0
                flags.append(valid and evidence.get("finite_gradients") is True)
                record = {"backend": backend, "so2cuda_call_count": calls,
                          "successful_so2cuda_call_count": successes,
                          "finite_gradients": evidence.get("finite_gradients") is True,
                          "dispatch_passed": valid}
                if "forward_backward_timer_verified" in evidence:
                    record["forward_backward_timer_verified"] = evidence["forward_backward_timer_verified"] is True
                dispatch.append(record)
            for key, child in node.items():
                visit(child, key if key in quantities else quantity)
        elif isinstance(node, list):
            for child in node:
                visit(child, quantity)
    visit(raw)
    if not rows or not flags:
        raise ValueError("Equivalence evidence needs numerical metrics and explicit pass decisions")
    passed = all(flags) and all(row.get("passed", True) and row.get("finite", True) for row in rows)
    if require_pass and not passed:
        raise ValueError("Numerical equivalence did not pass")
    return {"schema": "so2cuda-public-equivalence-v2", "all_passed": passed, "metric_count": len(rows),
            "max_abs": max(row["max_abs"] for row in rows), "max_relative_l2": max(row["relative_l2"] for row in rows),
            "metrics": rows, "backend_dispatch": dispatch}


def config_label(config):
    if "channels" in config:
        return f'lmax={config["lmax"]}, C={config["channels"]}, mmax={config["mmax"]}'
    pair = config["irreps_in"], config["irreps_out"]
    return {(DECREASING_2, DECREASING_2): "Decreasing channels, lmax=2",
            (DECREASING_4, DECREASING_4): "Decreasing channels, lmax=4",
            (V_SHAPE, V_SHAPE): "V-shaped channels", (V_SHAPE, TARGET): "V-shaped input → different output"}.get(pair, "Non-uniform channels")


def time_cell(row, accelerated=None, scope="forward_backward"):
    if row["status"] != "passed":
        return "OOM" if row["status"] == "oom" else "N/A"
    value = row[scope]["median_ms"]
    result = f"{value:.2f}"
    if accelerated is not None and accelerated["status"] == "passed":
        result += f' ({value / accelerated[scope]["median_ms"]:.2f}×)'
    return result


def spread_cell(data):
    if data is None:
        return "—"
    return f'{data["median_ms"]:.2f} ({data["q1_ms"]:.2f}–{data["q3_ms"]:.2f})'


def memory_cell(data):
    value = data.get("peak_allocated_bytes")
    return f"{value / 2 ** 30:.2f}" if value is not None else "—"


def operator_tables(report, groups):
    lines = []
    for group in groups:
        rows = sorted((r for r in report["cases"] if group in r["groups"]),
                      key=lambda r: (r["config"].get("lmax", 0), r["config"].get("channels", 0),
                                     r["config"]["irreps_in"], r["config"]["irreps_out"], r["config"]["edges"]))
        if group != "A4":
            lines.extend(["**" + GROUP_LABELS[group] + "**", ""])
        lines.extend(["| Configuration | Directed edges | " + " | ".join(LABELS[n] + " ms" + (" (ratio)" if n != "so2cuda" else "") for n in OPERATOR_COLUMNS) + " |",
                      "|---|---:|" + "---:|" * len(OPERATOR_COLUMNS)])
        for row in rows:
            impls = row["implementations"]
            cells = [time_cell(impls[n], impls["so2cuda"] if n != "so2cuda" else None) for n in OPERATOR_COLUMNS]
            lines.append(f'| {config_label(row["config"])} | {row["config"]["edges"]:,} | ' + " | ".join(cells) + " |")
        lines.append("")
    lines.extend(["<details>", "<summary>Forward timing, quartiles, peak memory, and cuEquivariance choices</summary>", "",
                  "| Configuration | Directed edges | Implementation | Forward ms (Q1–Q3) | Forward + backward ms (Q1–Q3) | Forward peak GiB | Forward + backward peak GiB |",
                  "|---|---:|---|---:|---:|---:|---:|"])
    included = [r for r in report["cases"] if any(g in r["groups"] for g in groups)]
    for row in included:
        for name in OPERATOR_COLUMNS:
            impl = row["implementations"][name]
            prefix = f'| {config_label(row["config"])} | {row["config"]["edges"]:,} | {LABELS[name]} | '
            if impl["status"] == "passed":
                lines.append(prefix + f'{spread_cell(impl["forward"])} | {spread_cell(impl["forward_backward"])} | {memory_cell(impl["forward"])} | {memory_cell(impl["forward_backward"])} |')
            else:
                marker = "OOM" if impl["status"] == "oom" else "N/A"
                lines.append(prefix + " | ".join([marker] * 4) + " |")
    lines.extend(["", "The cuEquivariance column uses the recorded descriptor, method, and rotation for each configuration:", "",
                  "| Configuration | Directed edges | Descriptor | Method | Rotation |", "|---|---:|---|---|---|"])
    for row in included:
        impl = row["implementations"]["cueq"]
        choice = ["`" + impl[k] + "`" if k in impl else "N/A" for k in ("descriptor", "method", "rotation")]
        lines.append(f'| {config_label(row["config"])} | {row["config"]["edges"]:,} | ' + " | ".join(choice) + " |")
    notes = sorted({LABELS[name] + ": " + impl["reason"] for row in included for name, impl in row["implementations"].items()
                    if impl["status"] != "passed"})
    if notes:
        lines.extend(["", *["- " + text for text in notes]])
    evidence_note = "Measured candidate timings and unsupported combinations are recorded in [operator JSON](docs/benchmarks/OP_SPEED_H200.json)."
    if any("cueq_selection_basis" in row for row in report["cases"]):
        evidence_note += " The original 12-configuration full scan is recorded in [selection evidence](docs/benchmarks/CUEQ_SELECTION_SCAN.json)."
    lines.extend(["", evidence_note, "", "</details>", ""])
    return "\n".join(lines)


def model_tables(report):
    lines = []
    for model, label in MODEL_LABELS.items():
        rows = [row for row in report["cases"] if row["model"] == model]
        unavailable = [row for row in report.get("unavailable_cases", []) if row["model"] == model]
        if unavailable:
            lines.extend(["**" + label + "**", "", "| Head | Measurement status | Reason |",
                          "|---|---|---|"])
            for row in unavailable:
                marker = "Failed" if row["status"] == "failed" else "Not measured"
                lines.append(f'| {row["head"].capitalize()} | {marker} | {row["reason"]} |')
            lines.append("")
            continue
        lines.extend(["**" + label + "**", "",
                      "| Head | Mean structures | Mean directed edges | " + " | ".join(LABELS[n] + " step ms" + (" (ratio)" if n != "so2cuda" else "") + " | " + LABELS[n] + " peak GiB" for n in MODEL_COLUMNS) + " |",
                      "|---|---:|---:|" + "---:|" * (2 * len(MODEL_COLUMNS))])
        for row in rows:
            impls, stream = row["implementations"], row["batch_stream"]
            cells = []
            for name in MODEL_COLUMNS:
                impl = impls[name]
                cells += [time_cell(impl, impls["so2cuda"] if name != "so2cuda" else None, "step"),
                          memory_cell(impl["step"]) if impl["status"] == "passed" else ("OOM" if impl["status"] == "oom" else "N/A")]
            lines.append(f'| {row["head"].capitalize()} | {stream["mean_structures"]:.2f} | {stream["mean_edges"]:,.1f} | ' + " | ".join(cells) + " |")
        lines.append("")
    if not report["cases"]:
        return "\n".join(lines)
    lines.extend(["<details>", "<summary>Step quartiles and forward + backward timing without the optimizer</summary>", "",
                  "| Model | Head | Backend | Step ms (Q1–Q3) | Forward + backward ms (Q1–Q3) | Peak allocated GiB |", "|---|---|---|---:|---:|---:|"])
    for row in report["cases"]:
        for name in MODEL_COLUMNS:
            impl = row["implementations"][name]
            prefix = f'| {MODEL_LABELS[row["model"]]} | {row["head"].capitalize()} | {LABELS[name]} | '
            if impl["status"] == "passed":
                lines.append(prefix + f'{spread_cell(impl["step"])} | {spread_cell(impl["forward_backward"])} | {memory_cell(impl["step"])} |')
            else:
                marker = "OOM" if impl["status"] == "oom" else "N/A"
                lines.append(prefix + " | ".join([marker] * 3) + " |")
    lines.extend(["", "</details>", ""])
    return "\n".join(lines)


def protocol(cases):
    warmups, iterations = sorted({r["warmup"] for r in cases}), sorted({r["iterations"] for r in cases})
    def span(values):
        return str(values[0]) if len(values) == 1 else f"{values[0]}–{values[-1]}"
    return span(warmups), span(iterations)


def so2cuda_version_sentence(operator, model):
    versions = {}
    for group in GROUP_LABELS:
        sha = operator["table_source_commits"][group]["SO2CUDA"]
        versions.setdefault(sha, []).append(group)
    measured_models = {row["model"] for row in model["cases"]}
    for name, label in MODEL_LABELS.items():
        if name not in measured_models:
            continue
        sha = model["table_source_commits"][name]["SO2CUDA"]
        versions.setdefault(sha, []).append(label)
    def commit_link(sha):
        return f"[`{sha[:8]}`](https://github.com/Franklalalala/SO2CUDA/tree/{sha})"
    uniform_shas = {operator["table_source_commits"][group]["SO2CUDA"] for group in ("A1", "A2", "A3")}
    if uniform_shas == {RELEASE_SHA} and set(versions) <= UNCHANGED_ROUTE_SHAS | {RELEASE_SHA}:
        other = [(sha, labels) for sha, labels in versions.items() if sha != RELEASE_SHA]
        prefix = "The uniform operator tables use SO2CUDA 0.3.2 source at " + commit_link(RELEASE_SHA)
        if not other:
            return prefix + ", as do all other measured tables."
        labels = {"A4": "the non-uniform operator table"}
        parts = [commit_link(sha) + " for " + ", ".join(labels.get(name, name) for name in names)
                 for sha, names in other]
        return prefix + "; the other measured tables use " + "; ".join(parts) + ", which execute the same kernels as 0.3.2 on these routes."
    if len(versions) == 1:
        return "All operator and model tables use SO2CUDA commit " + commit_link(next(iter(versions))) + "."
    parts = [commit_link(sha) + " for " + ", ".join(labels) for sha, labels in versions.items()]
    return "The SO2CUDA source commits are " + "; ".join(parts) + "."


def render(operator, model, equiv_op, equiv_model):
    op_warmup, op_iterations = protocol(operator["cases"])
    model_warmup, model_iterations = protocol(model["cases"]) if model["cases"] else (None, None)
    cueq_versions = {row["implementations"]["cueq"]["version"] for row in operator["cases"]
                     if "version" in row["implementations"]["cueq"]}
    if operator["versions"].get("cuequivariance"):
        cueq_versions.add(operator["versions"]["cuequivariance"])
    if len(cueq_versions) != 1:
        raise ValueError("Exactly one measured cuEquivariance version is required")
    cueq_version = next(iter(cueq_versions))
    fixed_choice = any("cueq_selection_basis" in row for row in operator["cases"])
    cueq_description = (
        "`SegmentedPolynomial` with `escn_tp_compact`, method `naive`, and precomputed PyTorch Wigner `bmm` rotation. "
        "All 16 descriptor/method/rotation combinations were compared on 12 uniform configurations; this combination "
        "was the fastest numerically correct choice in every configuration. The complete scan is retained in "
        "[selection evidence](docs/benchmarks/CUEQ_SELECTION_SCAN.json), and the tables use this fixed choice."
        if fixed_choice else
        "`SO3` irreps with `escn_tp` or `escn_tp_compact`, executed by `SegmentedPolynomial`. "
        "The selected method and rotation are listed below; alternatives and unsupported combinations are retained in the JSON. "
        "PyTorch rotation uses precomputed Wigner `bmm`; the `cueq` rotation choice uses the public `Rotation` interface.")
    nonuniform = [r for r in operator["cases"] if "A4" in r["groups"]]
    configurations = []
    for config in (r["config"] for r in nonuniform):
        label = config_label(config)
        if label not in {x[0] for x in configurations}:
            configurations.append((label, config["irreps_in"], config["irreps_out"]))
    model_protocol = (
        f"For each published model and head, all backends use the same structures in the same order, with {model_warmup} warmup steps and {model_iterations} measured steps on NVIDIA H200. "
        if model["cases"] else "No complete validated real-batch model table is available; no model timing or speedup is reported. ")
    model_protocol += (
        "We switch the SO(2) and associated expert-linear execution backend: default SO2CUDA, "
        "[our pure PyTorch implementation](examples/naive_baseline.py), or [cuEquivariance](examples/cueq_baseline.py). "
        "Parameters, routing, the remaining model, and the loss stay fixed. HybridMuon uses the same fast optimizer path in every case. "
        "Published timing tables show the median complete step (forward + backward + optimizer), its ratio to SO2CUDA, and peak allocated memory. "
        "Forward + backward timing excluding the optimizer appears in the details.")
    if model.get("status") == "incomplete":
        model_protocol += " Failed or unmeasured tables are disclosed below; their raw evidence digests are retained in the public JSON."
    validation_sentence = (
        "For each published timing table, the first-step loss and parameter gradients are compared on the same batch within FP32 rounding tolerance, "
        "and dispatch checks verify that the pure PyTorch and cuEquivariance routes do not call SO2CUDA. "
        if model["cases"] else "Formal real-batch model equivalence is unavailable for these tables. ")
    if model.get("status") == "incomplete" and model["cases"]:
        validation_sentence += "No equivalence pass is claimed for the unavailable tables. "
    lines = [BEGIN, "## Performance", "", "### 1. SO(2) tensor product operator", "",
             "The edge operator rotates features into the edge frame, applies a shared linear map for each |m|, and rotates back:", "",
             r"$$y_e = D(R_e)^{\top}\,\mathcal{L}_W\!\left(D(R_e)\,x_e\right).$$", "",
             "The m=0 block is real; each positive m uses a complex linear map represented by real weights. Weights are shared across edges; the operator benchmark excludes radial modulation.", "",
             "**Uniform irreps** have the same number of channels at every l, as in the usual eSCN, EquiformerV2, and EquiformerV3 layouts. **Non-uniform irreps** have different channel counts across l. SO2CUDA and our pure PyTorch implementation support both, including unequal input and output irreps. The original EquiformerV3 layer requires uniform channels on each side; non-uniform cases are N/A without padding. cuEquivariance support depends on the descriptor and method; unsupported combinations retain their reason in the JSON.", "",
             "- **SO2CUDA:** the public `true_dense_pairs(..., include_m0=True)` interface, with default settings, computes the complete layer.",
             "- **Our pure PyTorch:** our implementation of [DeePTB upstream `SO2_Linear`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py), in [operator_baselines.py](examples/operator_baselines.py). It is authored here from that computation pattern.",
             f"- **EquiformerV3:** the original `SO3Rotation` and `SO2Linear` at [commit `{EQV3_SHA[:8]}`](https://github.com/atomicarchitects/equiformer_v3/tree/{EQV3_SHA}), with m=0 bias set to zero. Eager and `torch.compile(dynamic=True)` are measured separately; compilation is outside steady-state timing.",
             f"- **cuEquivariance {cueq_version}:** " + cueq_description, "",
             f"Measurements use NVIDIA H200 in strict FP32 with TF32 disabled. {so2cuda_version_sentence(operator, model)} Geometry and feature-layout conversions are prepared before timing in each implementation's native format. Each implementation has {op_warmup} warmup iterations and {op_iterations} measured iterations. Each complete table comes from one card task. The main tables show forward + backward medians in ms, including input and weight gradients. Ratios are the comparison time divided by SO2CUDA time: above 1 means SO2CUDA is faster; below 1 means the comparison is faster. Peak allocated memory includes that implementation's inputs, weights, geometry, saved activations, gradients, and workspace.", "",
             "#### 1.1 Uniform irreps", "", operator_tables(operator, ("A1", "A2", "A3")),
             "#### 1.2 Non-uniform irreps", "", "The configurations below use decreasing channels, UniTB's V-shaped hidden channels, and unequal input/output irreps:", "",
             "| Configuration | Input irreps | Output irreps |", "|---|---|---|"]
    for label, ir_in, ir_out in configurations:
        lines.append(f"| {label} | `{ir_in}` | `{ir_out}` |")
    lines.extend(["", operator_tables(operator, ("A4",)),
                  "Outputs, input gradients, and weight gradients mapped to common canonical parameters are compared pairwise and against an FP64 reference on small inputs. The new non-uniform irreps also undergo a rotation-equivariance check. [Numerical evidence](docs/benchmarks/EQUIV_OP_L40S.json) records the errors and FP32 tolerances.", "",
                  "**Minimal use for interatomic potentials**", "",
                  "The example accepts e3nn irreps such as `32x0e+32x1o+32x2e` or `128x0e+64x1o+32x2e`, and permits different input/output channel counts. Features use e3nn `mul_ir` layout `[edges, irreps.dim]`. Edge vectors determine a constant rotation; geometry gradients are unsupported. Prepare the layout and Wigner data once for an unchanged edge geometry:", "", "```python",
                  "import torch", "from e3nn import o3", "from examples.minimal_so2_tp import prepare_so2_tp, make_weights",
                  "from so2_cuda_ops.deeptb import true_dense_pairs", "", 'irreps = o3.Irreps("128x0e+64x1o+32x2e")',
                  'edge_vectors = torch.randn(512, 3, device="cuda", dtype=torch.float32)',
                  'x = torch.randn(512, irreps.dim, device="cuda", dtype=torch.float32, requires_grad=True)',
                  "layout, wigner = prepare_so2_tp(irreps, irreps, edge_vectors, m_max=2)",
                  "weights = make_weights(irreps, irreps, 2, device=x.device)",
                  "parts = None if wigner is None else true_dense_pairs(x, layout, wigner, weights, include_m0=True)",
                  'if parts is None:', '    raise RuntimeError("Use your PyTorch reference implementation for this input")',
                  "y = parts[0]", "y.square().mean().backward()", "```", "",
                  "`make_weights` creates trainable W0 and `[A_m; B_m]` matrices, shared across edges; a model can supply its own `LinearWeights` instead. These preparation helpers live in [minimal_so2_tp.py](examples/minimal_so2_tp.py), rather than the package API. The CUDA interface uses FP32 tensors on one CUDA device. CPU, other dtypes, autocast, `torch.func`, geometry requiring gradients, and unsupported layouts return `None`, so the caller can use its reference path. Native CUDA execution errors propagate.", "",
                  "Reproduce an operator comparison from the repository root. The environment must already contain CUDA-enabled PyTorch, compatible cuBLAS, and the standard Python dependencies (NumPy, SciPy, SymPy, NetworkX, opt-einsum, tqdm, nvidia-ml-py, and platformdirs). `--no-deps` preserves that environment; measured software versions are recorded in the JSON:", "", "```bash",
                  f"pip install --no-deps cuequivariance=={cueq_version} cuequivariance-torch=={cueq_version} cuequivariance-ops-cu12=={cueq_version} cuequivariance-ops-torch-cu12=={cueq_version}",
                  f"git clone https://github.com/atomicarchitects/equiformer_v3.git && git -C equiformer_v3 checkout {EQV3_SHA}",
                  "python examples/so2_operator_speed_test.py --impl naive,so2cuda,eqv3,cueq --include-compile --suite all --eqv3-root equiformer_v3 --warmup 5 --iterations 20" +
                  (" --cueq-choice escn_tp_compact,naive,pytorch" if fixed_choice else "") + " --json operator.json",
                  "```", "", "### 2. UniTB models on real training batches", "",
                  "UniTB uses PDQ-MoE; UniTB-dense uses a single expert; UniTB-SLEM applies three SO(2) operators per layer. The comparison uses both onsite and hopping production configurations without model changes. Batches contain real crystal structures from our training set, with a limit of 32 structures per batch. Dynamic cost limits can produce smaller batches; published timing tables report the measured mean structure and directed-edge counts for each fixed batch stream.", "",
                  model_protocol, "",
                  f"The DeePTB source is pinned to [commit `{DEEPTB_SHA[:7]}`](https://github.com/Franklalalala/DeePTB/tree/{DEEPTB_SHA}); full source commits and software versions are recorded in the public JSON.", "",
                  model_tables(model),
                  "EquiformerV3 is N/A for all three complete models because each contains non-uniform SO(2) layers, including UniTB-dense's final output layers.", "",
                  "The pure PyTorch model baseline is our implementation of the DeePTB upstream SO(2) computation and [UMA MoLE linear formula](https://github.com/facebookresearch/fairchem/blob/3801dac0cc0458a2f8121259a2ce8b23d4dcc5a1/src/fairchem/core/models/uma/nn/mole.py). " + validation_sentence + "See [model measurement records](docs/benchmarks/MODEL_BS32_H200.json) and [model equivalence](docs/benchmarks/EQUIV_MODEL.json).", "",
                  "The training dataset is not public. Synthetic periodic structures provide a runnable comparison of relative timing; they are used only for timing and do not reproduce the real-batch measurements, physical priors, or prediction accuracy:", "", "```bash",
                  "pip install 'git+https://github.com/Franklalalala/DeePTB.git@1006-stable'", "pip install -e .",
                  "python examples/deeptb_speed_test.py --model all --backend both --edges 20000 --json synthetic_so2cuda.json",
                  "python examples/deeptb_speed_test.py --model all --backend cueq --edges 20000 --json synthetic_cueq.json", "```", "",
                  "The synthetic example compares the same randomly initialized model and inputs, disables SO2CUDA for reference routes, and records timing, peak memory, dispatch, and numerical differences. Reduce `--edges` on smaller GPUs; `--max-memory-gib` sets its allocator limit. Its forward + backward timings exclude optimizer updates.", "", END])
    return "\n".join(lines) + "\n"


def replace_section(readme, section):
    if BEGIN not in readme or END not in readme:
        raise ValueError("README benchmark markers are missing")
    start = readme.index(BEGIN)
    finish = readme.index(END, start) + len(END)
    return readme[:start] + section.rstrip() + readme[finish:]


def verify_sources(operator, model):
    op_tables = table_source_commits(operator["cases"], operator, operator=True)
    pinned_models = model["cases"] + [row for row in model.get("unavailable_cases", [])
                                      if "SO2CUDA" in row["source_commits"]]
    model_tables = table_source_commits(pinned_models, model, operator=False)
    for commits in op_tables.values():
        if set(commits) != {"SO2CUDA", "DeePTB", "EquiformerV3"}:
            raise ValueError("Every operator table must pin SO2CUDA, DeePTB, and EquiformerV3")
        if commits["EquiformerV3"] != EQV3_SHA:
            raise ValueError("EquiformerV3 benchmark used another source commit")
    for commits in model_tables.values():
        if not {"SO2CUDA", "DeePTB"} <= set(commits):
            raise ValueError("Every model/head result must pin SO2CUDA and DeePTB")
    for commits in (*op_tables.values(), *model_tables.values()):
        if any(not re.fullmatch(r"[0-9a-f]{40}", value) for value in commits.values()):
            raise ValueError("Source commits must be full forty-character SHA values")
        if commits["DeePTB"] != DEEPTB_SHA:
            raise ValueError("Real-batch measurements must use the pinned DeePTB release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator-json", type=Path, required=True)
    parser.add_argument("--model-json", type=Path, required=True)
    parser.add_argument("--equiv-operator-json", type=Path, required=True)
    parser.add_argument("--equiv-model-json", type=Path, required=True)
    parser.add_argument("--allow-incomplete-models", action="store_true",
                        help="Publish only validated whole model tables and explicitly disclose failed or unmeasured tables")
    parser.add_argument("--cueq-selection-json", type=Path,
                        help="Original full 12-configuration cuEquivariance scan (otherwise use the aggregate's embedded scan)")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--readme", type=Path, default=root / "README.md")
    parser.add_argument("--docs-dir", type=Path, default=root / "docs/benchmarks")
    parser.add_argument("--write-readme", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check README and public JSON without writing files")
    args = parser.parse_args()
    if args.check and args.write_readme:
        parser.error("--check and --write-readme are mutually exclusive")
    raw_operator = load(args.operator_json)
    operator = operator_evidence(raw_operator)
    selection_scan = selection_scan_artifact(raw_operator, args.cueq_selection_json)
    model = model_evidence(load(args.model_json), allow_incomplete=args.allow_incomplete_models)
    verify_sources(operator, model)
    equiv_op = equivalence_evidence(load(args.equiv_operator_json))
    equiv_model = model_equivalence_evidence(load(args.equiv_model_json), model)
    inputs = {"operator": digest(args.operator_json), "model": digest(args.model_json),
              "equiv_operator": digest(args.equiv_operator_json), "equiv_model": digest(args.equiv_model_json)}
    if selection_scan is not None:
        inputs["cueq_selection_scan"] = selection_scan["selection_basis"]["source_sha256"]
    section = render(operator, model, equiv_op, equiv_model)
    artifacts = {"OP_SPEED_H200": operator, "MODEL_BS32_H200": model,
                 "EQUIV_OP_L40S": equiv_op, "EQUIV_MODEL": equiv_model}
    if selection_scan is not None:
        artifacts["CUEQ_SELECTION_SCAN"] = selection_scan
    expected = {args.docs_dir / (name + ".json"): json.dumps({**value, "input_sha256": inputs}, indent=2,
                ensure_ascii=False, allow_nan=False) + "\n" for name, value in artifacts.items()}
    expected[args.docs_dir / "README_SECTION.md"] = section
    readme = args.readme.read_text(encoding="utf-8")
    generated_readme = replace_section(readme, section)
    if args.check:
        differences = [str(path.name) for path, value in expected.items()
                       if not path.is_file() or path.read_text(encoding="utf-8") != value]
        if generated_readme != readme:
            differences.append(args.readme.name)
        if differences:
            raise SystemExit("Generated artifacts differ: " + ", ".join(differences))
        print(json.dumps({"status": "passed", "readme_matches_json": True}))
        return
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    for path, value in expected.items():
        path.write_text(value, encoding="utf-8")
    if args.write_readme:
        args.readme.write_text(generated_readme, encoding="utf-8")
    print(json.dumps({"public_json_files": len(artifacts), "readme_updated": args.write_readme}))


if __name__ == "__main__":
    main()
