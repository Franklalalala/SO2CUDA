"""CPU publication checks; synthetic timings here are never benchmark evidence."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import pytest

from examples import update_readme_benchmarks as gen


BASE_SHA = "a" * 40
OTHER_SHA = "b" * 40
COMMITS = {"SO2CUDA": BASE_SHA, "DeePTB": gen.DEEPTB_SHA, "EquiformerV3": gen.EQV3_SHA}
EQUIVALENCE = {"passed": True, "output": {"max_abs": 1e-6, "relative_l2": 1e-7,
                                                     "passed": True, "finite": True}}


def timed(value=1., *, model=False):
    statistics = {"median_ms": value, "q1_ms": value, "q3_ms": value,
                  "samples_ms": [value] * (12 if model else 20), "peak_allocated_bytes": 1024}
    row = {"status": "passed", "forward_backward": deepcopy(statistics),
           "exclusive_gpu_proof": [{"exclusive": True}, {"exclusive": True}]}
    row["step" if model else "forward"] = deepcopy(statistics)
    if model:
        row.update(optimizer="HybridMuon", optimizer_mode="fast")
    return row


def cueq_records():
    selected = {**timed(), **gen.CUEQ_CHOICE, "version": "0.12.0", "equivalence": EQUIVALENCE}
    candidates = []
    for descriptor, method, rotation in itertools.product(sorted(gen.DESCRIPTORS), sorted(gen.METHODS), ("pytorch", "cueq")):
        choice = dict(descriptor=descriptor, method=method, rotation=rotation)
        candidates.append(deepcopy(selected) if choice == gen.CUEQ_CHOICE else
                          {**timed(2.), **choice, "version": "0.12.0"})
    return selected, candidates


@pytest.fixture
def reports():
    rows = []
    for group, configurations in gen.expected_operator_tables().items():
        for ir_in, ir_out, mmax, edges in sorted(configurations):
            chosen, alternatives = cueq_records()
            implementations = {name: timed() for name in gen.OPERATOR_COLUMNS}
            implementations["cueq"] = chosen
            implementations["so2cuda"]["metadata"] = {"api": "so2_cuda_ops.deeptb.true_dense_pairs"}
            if group == "A4":
                for name in ("eqv3", "eqv3+compile"):
                    implementations[name] = {"status": "N/A", "reason": "Uniform channels required"}
            else:
                counters = dict(unique_graphs=1, calls_captured=1, frames_ok=1, aot_autograd_ok=1)
                implementations["eqv3+compile"]["compile_execution"] = {
                    "executed": True, "counter_delta_before_timing": counters,
                    "counter_delta_after_timing": counters}
            rows.append({"groups": [group], "status": "completed", "task_id": group + "-task",
                         "config": dict(irreps_in=ir_in, irreps_out=ir_out, mmax=mmax, edges=edges),
                         "implementations": implementations, "cueq_alternatives": alternatives})
    operator = {"status": "completed", "gpu": "NVIDIA H200", "precision": "strict FP32; TF32 disabled",
                "warmup": 5, "iterations": 20, "commits": COMMITS, "cases": rows}
    models = {"status": "completed", "cases": [
        {"model": model, "head": head, "task_id": model + "-task", "gpu": "NVIDIA H200",
         "warmup_steps": 4, "measured_steps": 12, "precision": "strict FP32; TF32 disabled",
         "source_commits": deepcopy(COMMITS), "batch_stream": dict(mean_structures=32, mean_edges=50000, sha256="c" * 64),
         "backends": {backend: timed(model=True) for backend in gen.MODEL_COLUMNS}}
        for model in gen.MODEL_LABELS for head in ("onsite", "hopping")]}
    return operator, models


def with_selection(operator):
    operator = deepcopy(operator)
    scan_rows = [deepcopy(row) for row in operator["cases"] if row["groups"] == ["A1"]]
    configs = {gen.operator_configuration(row["config"]) for row in scan_rows}
    extra = [deepcopy(row) for row in operator["cases"] if row["groups"] == ["A2"]
             and gen.operator_configuration(row["config"]) not in configs]
    scan_rows.extend(extra[:3])
    for row in scan_rows:
        row["task_id"] = "selection-task"
        row["implementations"]["cueq"]["metadata"] = {"file": "/private/source/cuEq.py"}
    scan = {**deepcopy(operator), "status": "running", "cases": scan_rows}
    scan_bytes = json.dumps(scan).encode()
    basis = {"schema": "so2-cueq-selection-basis-v1", "choice": gen.CUEQ_CHOICE,
             "source_sha256": hashlib.sha256(scan_bytes).hexdigest(), "source_commits": COMMITS,
             "evidence_file": gen.CUEQ_SELECTION_FILE, "config_count": 12,
             "alternatives_per_config": 16, "metric": "forward_backward.median_ms"}
    for row in operator["cases"]:
        if row["groups"] != ["A1"]:
            row["cueq_alternatives"] = [deepcopy(row["implementations"]["cueq"])]
            row["cueq_selection_basis"] = deepcopy(basis)
    operator.update(cueq_selection_basis=basis, cueq_selection_scan=scan,
                    cueq_selection_session={"raw_sha256": basis["source_sha256"]})
    return operator, scan_bytes


def test_selected_choice_preserves_original_scan(reports):
    raw, _ = with_selection(reports[0])
    public = gen.operator_evidence(raw)
    assert all(len(row["cueq_alternatives"]) == (16 if row["groups"] == ["A1"] else 1)
               for row in public["cases"])
    scan = gen.selection_scan_artifact(raw)
    assert len(scan["cases"]) == 12
    assert sum(len(row["cueq_alternatives"]) for row in scan["cases"]) == 192
    assert all(row["selected_equivalence"]["all_passed"] for row in scan["cases"])
    assert "/private" not in json.dumps(scan)
    assert "selection-task" not in json.dumps(scan)


@pytest.mark.parametrize("mutation", [
    lambda raw: (raw["cases"][-1].pop("cueq_selection_basis"), raw.pop("cueq_selection_basis")),
    lambda raw: raw["cases"][-1]["cueq_selection_basis"].update(source_sha256="invalid"),
    lambda raw: raw["cases"][-1]["cueq_alternatives"][0].update(method="uniform_1d"),
    lambda raw: raw["cases"][-1]["implementations"]["cueq"]["forward_backward"].update(median_ms=2., q3_ms=2., samples_ms=[2.] * 20),
])
def test_invalid_selected_evidence_rejected(reports, mutation):
    raw, _ = with_selection(reports[0])
    mutation(raw)
    with pytest.raises(ValueError):
        gen.operator_evidence(raw)


def test_selected_unsupported_shape_keeps_choice(reports):
    raw, _ = with_selection(reports[0])
    row = raw["cases"][-1]
    row["implementations"]["cueq"] = {"status": "N/A", "reason": "Unsupported channel layout"}
    row["cueq_alternatives"] = [{"status": "N/A", "reason": "Unsupported channel layout", **gen.CUEQ_CHOICE}]
    public = gen.operator_evidence(raw)
    assert public["cases"][-1]["implementations"]["cueq"]["descriptor"] == "escn_tp_compact"


def test_scan_hash_and_full_grid_are_required(reports, tmp_path):
    raw, data = with_selection(reports[0])
    path = tmp_path / "scan.json"
    path.write_bytes(data)
    assert gen.selection_scan_artifact(raw, path) == gen.selection_scan_artifact(raw)
    path.write_bytes(data + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        gen.selection_scan_artifact(raw, path)
    raw["cueq_selection_scan"]["cases"][0]["cueq_alternatives"].pop()
    with pytest.raises(ValueError, match="candidate records"):
        gen.selection_scan_artifact(raw)


def test_mixed_commits_are_reported_per_table(reports):
    operator, model = deepcopy(reports)
    for row in operator["cases"]:
        if row["groups"] != ["A4"]:
            row["source_commits"] = {**COMMITS, "SO2CUDA": OTHER_SHA}
    for row in model["cases"]:
        if row["model"] == "dense":
            row["source_commits"]["SO2CUDA"] = OTHER_SHA
    op, models = gen.operator_evidence(operator), gen.model_evidence(model)
    gen.verify_sources(op, models)
    assert "SO2CUDA" not in op["source_commits"]
    sentence = gen.so2cuda_version_sentence(op, models)
    assert "for A1, A2, A3, UniTB-dense" in sentence
    assert "for A4, UniTB, UniTB-SLEM" in sentence
    assert sentence.count("The SO2CUDA source commits") == 1


@pytest.mark.parametrize("kind", ["operator", "model"])
def test_mixed_commits_inside_table_rejected(reports, kind):
    operator, model = deepcopy(reports)
    if kind == "operator":
        operator["cases"][0]["source_commits"] = {**COMMITS, "SO2CUDA": OTHER_SHA}
        action = lambda: gen.operator_evidence(operator)
    else:
        model["cases"][0]["source_commits"]["SO2CUDA"] = OTHER_SHA
        action = lambda: gen.model_evidence(model)
    with pytest.raises(ValueError, match="different source commits"):
        action()


def test_cli_keeps_scan_and_reproduces_all_artifacts(reports, tmp_path):
    operator, model = reports
    operator, _ = with_selection(operator)
    for name, value in (("operator", operator), ("model", model), ("equivalence", EQUIVALENCE)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    readme = tmp_path / "README.md"
    readme.write_text(gen.BEGIN + "\n" + gen.END + "\n")
    command = [sys.executable, str(Path(gen.__file__)), "--operator-json", str(tmp_path / "operator.json"),
               "--model-json", str(tmp_path / "model.json"), "--equiv-operator-json", str(tmp_path / "equivalence.json"),
               "--equiv-model-json", str(tmp_path / "equivalence.json"), "--readme", str(readme),
               "--docs-dir", str(tmp_path / "public")]
    subprocess.run(command + ["--write-readme"], check=True, capture_output=True)
    subprocess.run(command + ["--check"], check=True, capture_output=True)
    assert (tmp_path / "public/CUEQ_SELECTION_SCAN.json").is_file()
    assert "All 16 descriptor/method/rotation combinations" in readme.read_text()
    assert "--cueq-choice escn_tp_compact,naive,pytorch" in readme.read_text()
    (tmp_path / "public/CUEQ_SELECTION_SCAN.json").write_text("{}")
    result = subprocess.run(command + ["--check"], capture_output=True, text=True)
    assert result.returncode != 0 and "CUEQ_SELECTION_SCAN.json" in result.stderr


def incomplete_models(raw, missing=("dense", "unitb", "slem")):
    raw = deepcopy(raw)
    unavailable = []
    for row in raw["cases"]:
        if row["model"] in missing:
            unavailable.append({key: row[key] for key in ("model", "head", "task_id", "source_commits")})
            unavailable[-1].update(status="failed", reason_code="progress_timeout",
                                   reason="Failure in /private/account/runtime with host=private-machine",
                                   raw_evidence_sha256={"RESULT.json": "d" * 64})
    raw.update(status="incomplete", unavailable_cases=unavailable,
               cases=[row for row in raw["cases"] if row["model"] not in missing])
    equivalence = {"status": "incomplete", "cases": [
        {"model": row["model"], "head": row["head"], "equivalence": deepcopy(EQUIVALENCE)} for row in raw["cases"]],
        "unavailable_cases": [{key: row[key] for key in ("model", "head", "status")} for row in unavailable]}
    return raw, equivalence


def test_incomplete_publication_requires_explicit_option_and_keeps_failure(reports):
    raw, equiv = incomplete_models(reports[1])
    with pytest.raises(ValueError, match="has not completed"):
        gen.model_evidence(raw)
    public = gen.model_evidence(raw, allow_incomplete=True)
    assert not public["cases"] and len(public["unavailable_cases"]) == 6
    assert all(row["status"] == "failed" for row in public["unavailable_cases"])
    assert public["unavailable_cases"][0]["raw_evidence_sha256"] == {"RESULT.json": "d" * 64}
    assert "private" not in json.dumps(public) and "dense-task" not in json.dumps(public)
    validation = gen.model_equivalence_evidence(equiv, public)
    assert validation["all_passed"] is False and validation["metric_count"] == 0
    section = gen.render(gen.operator_evidence(reports[0]), public, EQUIVALENCE, validation)
    assert "No complete validated real-batch model table is available" in section
    assert "Formal real-batch model equivalence is unavailable" in section
    assert "| Onsite | Failed |" in section
    assert "None warmup" not in section


def test_optional_publication_rejects_half_model_table(reports):
    raw, _ = incomplete_models(reports[1], ("dense",))
    raw["unavailable_cases"] = [row for row in raw["unavailable_cases"] if row["head"] == "hopping"]
    raw["cases"].append(deepcopy(reports[1]["cases"][0]))
    with pytest.raises(ValueError, match="both heads"):
        gen.model_evidence(raw, allow_incomplete=True)


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["unavailable_cases"][0].update(raw_evidence_sha256={}),
    lambda raw: raw["unavailable_cases"][0].update(raw_evidence_sha256={"/private/RESULT.json": "d" * 64}),
    lambda raw: raw["unavailable_cases"][0].update(status="N/A"),
    lambda raw: raw["unavailable_cases"][0].pop("task_id"),
])
def test_unavailable_failure_requires_evidence_not_na(reports, mutation):
    raw, _ = incomplete_models(reports[1])
    mutation(raw)
    with pytest.raises(ValueError):
        gen.model_evidence(raw, allow_incomplete=True)


@pytest.mark.parametrize("mutation", [
    lambda row: row["backends"]["so2cuda"].update(status="failed"),
    lambda row: row["backends"]["cueq"].update(batch_stream_sha256="e" * 64),
    lambda row: row["backends"]["naive"].update(optimizer="other"),
    lambda row: row["backends"]["so2cuda"]["step"].update(median_ms=2.),
])
def test_optional_publication_keeps_success_validation_strict(reports, mutation):
    raw, _ = incomplete_models(reports[1], ("dense", "unitb"))
    mutation(raw["cases"][0])
    with pytest.raises(ValueError):
        gen.model_evidence(raw, allow_incomplete=True)


def test_partial_equivalence_is_scoped_to_validated_tables(reports):
    raw, equiv = incomplete_models(reports[1], ("dense", "unitb"))
    raw["cases"][0]["backends"]["cueq"] = {"status": "oom"}
    raw["cases"][1]["backends"]["cueq"] = {"status": "N/A", "reason": "Unsupported configuration"}
    public = gen.model_evidence(raw, allow_incomplete=True)
    assert public["cases"][0]["implementations"]["cueq"]["status"] == "oom"
    assert public["cases"][1]["implementations"]["cueq"]["status"] == "N/A"
    validation = gen.model_equivalence_evidence(equiv, public)
    assert validation["metric_count"] == 2 and not validation["all_passed"]
    equiv["cases"].append({"model": "dense", "head": "onsite", "equivalence": EQUIVALENCE})
    with pytest.raises(ValueError, match="exactly the validated"):
        gen.model_equivalence_evidence(equiv, public)


def test_release_sentence_pins_actual_measurements_and_shared_kernels(reports):
    operator, models = deepcopy(reports)
    for row in operator["cases"]:
        sha = next(iter(gen.UNCHANGED_ROUTE_SHAS)) if row["groups"] == ["A4"] else gen.RELEASE_SHA
        row["source_commits"] = {**COMMITS, "SO2CUDA": sha}
    models, _ = incomplete_models(models)
    sentence = gen.so2cuda_version_sentence(gen.operator_evidence(operator),
                                          gen.model_evidence(models, allow_incomplete=True))
    assert "SO2CUDA 0.3.2 source" in sentence and "same kernels as 0.3.2" in sentence
    assert "UniTB" not in sentence


def test_incomplete_cli_generation_is_reproducible(reports, tmp_path):
    model, equivalent = incomplete_models(reports[1])
    for name, value in (("operator", reports[0]), ("model", model),
                        ("equivalence", EQUIVALENCE), ("model_equivalence", equivalent)):
        (tmp_path / (name + ".json")).write_text(json.dumps(value))
    readme = tmp_path / "README.md"
    readme.write_text(gen.BEGIN + "\n" + gen.END + "\n")
    command = [sys.executable, str(Path(gen.__file__)), "--operator-json", str(tmp_path / "operator.json"),
               "--model-json", str(tmp_path / "model.json"), "--equiv-operator-json", str(tmp_path / "equivalence.json"),
               "--equiv-model-json", str(tmp_path / "model_equivalence.json"), "--readme", str(readme),
               "--docs-dir", str(tmp_path / "public"), "--allow-incomplete-models"]
    subprocess.run(command + ["--write-readme"], check=True, capture_output=True)
    subprocess.run(command + ["--check"], check=True, capture_output=True)
    public = json.loads((tmp_path / "public/EQUIV_MODEL.json").read_text())
    assert public["status"] == "incomplete" and not public["all_passed"]
