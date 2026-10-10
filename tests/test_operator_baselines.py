"""Numerical contracts for the optional original/public operator baselines."""
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
from e3nn import o3

_examples = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(_examples))
from operator_baselines import (
    ActivationOperator, Geometry, CueqOperator, NaiveOperator, SO2CUDAOperator, ExplicitGEMMOperator,
    canonical_weights, prepare_geometry, uniform_irreps,
)
from operator_eqv3 import Eqv3Operator
from so2_operator_speed_test import metric
import so2_operator_speed_test as speed


def test_operator_adapters_import_without_optional_dependencies(monkeypatch):
    """Importing examples must not require a third-party checkout or CUDA."""
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "cuequivariance", None)
        patch.setitem(sys.modules, "cuequivariance_torch", None)
        for name in ("operator_baselines", "operator_cueq", "operator_eqv3"):
            importlib.reload(importlib.import_module(name))


def _snapshot(operator, x, cotangent):
    operator.zero_grad(set_to_none=True)
    source = operator.input_to_native(x.detach()).detach().requires_grad_()
    result = operator.forward_native(source)
    result.backward(operator.output_to_native(cotangent.to(result.dtype)))
    gradients = operator.canonical_gradients()
    assert all(gradient is not None for gradient in gradients)
    return (operator.output_from_native(result.detach()),
            operator.input_from_native(source.grad.detach()),
            *(gradient.detach() for gradient in gradients))


def _assert_snapshot(actual, expected):
    assert len(actual) == len(expected)
    for got, ref in zip(actual, expected):
        got, ref = got.double(), ref.double()
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, atol=4e-5, rtol=2e-4)
        relative = (got - ref).norm() / ref.norm().clamp_min(1e-12)
        assert relative < 3e-5


def _assert_equivariance(factory, irreps_in, irreps_out, weights, vectors, x, mmax):
    """Use independently evaluated float64 representation matrices."""
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        q = o3.angles_to_matrix(*[torch.tensor(angle) for angle in (.37, 1.11, -.51)])
        dx = irreps_in.D_from_matrix(q).to(device=x.device, dtype=x.dtype)
        dy = irreps_out.D_from_matrix(q).to(device=x.device, dtype=x.dtype)
    finally:
        torch.set_default_dtype(previous_dtype)
    q = q.to(device=x.device, dtype=x.dtype)
    lmax = max(irreps_in.lmax, irreps_out.lmax)
    first = factory(irreps_in, irreps_out, mmax, weights, prepare_geometry(vectors, lmax))
    second = factory(irreps_in, irreps_out, mmax, weights,
                     prepare_geometry(vectors @ q.T, lmax))
    with torch.no_grad():
        check = metric(second(x @ dx.T), first(x) @ dy.T)
    assert check["passed"], check


def test_canonical_complex_pair_signs_and_gradients_on_cpu():
    """An identity frame isolates the +/-m sign convention from rotation."""
    irreps = uniform_irreps(1, 2)
    vectors = torch.tensor([[0., 1., 0.], [0., 1., 0.]])
    geometry = prepare_geometry(vectors, 1)
    m0 = torch.zeros(4, 4)
    pair = torch.tensor([[1., 2.], [-3., 4.], [5., -6.], [7., 8.]])
    operator = NaiveOperator(irreps, irreps, 1, (m0, pair), geometry)
    x = torch.tensor([[0., 0., 1., 2., 3., 4., 5., 6.],
                      [0., 0., -1., 2., -3., 4., -5., 6.]], requires_grad=True)
    actual = operator(x)
    a, b = operator.weights[1].chunk(2, dim=0)
    minus, plus = x[:, [2, 5]], x[:, [4, 7]]
    expected = torch.zeros_like(actual)
    expected[:, [2, 5]] = minus @ a.T - plus @ b.T
    expected[:, [4, 7]] = plus @ a.T + minus @ b.T
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    cotangent = torch.linspace(-0.6, 0.9, actual.numel()).reshape_as(actual)
    targets = (x, operator.weights[1])
    actual_gradient = torch.autograd.grad(actual, targets, cotangent, retain_graph=True)
    expected_gradient = torch.autograd.grad(expected, targets, cotangent)
    for got, ref in zip(actual_gradient, expected_gradient):
        torch.testing.assert_close(got, ref, atol=1e-6, rtol=1e-6)


_SHAPES = (
    ("3x0e+3x1o+3x2e", "3x0e+3x1o+3x2e", 2),
    ("2x0e+2x1o+2x2e+2x3o+2x4e", "3x0e+3x1o+3x2e+3x3o+3x4e", 2),
    ("3x2e+2x0e+1x1o", "2x1o+1x2e+3x0e", 2),
)


@pytest.mark.parametrize("irreps_in,irreps_out,mmax", _SHAPES)
@pytest.mark.parametrize("implementation,descriptor_name", [
    ("so2cuda", None), ("activation", None), ("explicit_gemm", None), ("cueq", "escn_tp"),
    ("cueq", "escn_tp_compact"), ("eqv3", None),
])
def test_available_baselines_match_float64_reference(
        implementation, descriptor_name, irreps_in, irreps_out, mmax):
    if not torch.cuda.is_available():
        pytest.skip("Operator forward and gradient comparisons require CUDA")
    if implementation == "cueq":
        pytest.importorskip("cuequivariance")
        pytest.importorskip("cuequivariance_torch")
    eqv3_root = os.environ.get("SO2CUDA_EQV3_ROOT")
    if implementation == "eqv3":
        if not eqv3_root:
            pytest.skip("Set SO2CUDA_EQV3_ROOT to the pinned original checkout")
        if len({mul for mul, _ in o3.Irreps(irreps_in)}) != 1:
            pytest.skip("Original EquiformerV3 cannot express nonuniform channels")
    if implementation == "explicit_gemm" and len({mul for mul, _ in o3.Irreps(irreps_in)}) != 1:
        pytest.skip("The explicit-rotation ablation covers uniform channels")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(719)
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    lmax = max(irreps_in.lmax, irreps_out.lmax)
    weights = canonical_weights(irreps_in, irreps_out, mmax, device="cuda")
    vectors = torch.randn(41, 3, device="cuda")
    geometry = prepare_geometry(vectors, lmax)
    x = torch.randn(41, irreps_in.dim, device="cuda")
    cotangent = torch.randn(41, irreps_out.dim, device="cuda")
    reference = NaiveOperator(irreps_in, irreps_out, mmax,
                              tuple(weight.double() for weight in weights),
                              prepare_geometry(vectors.double(), lmax))
    if implementation == "so2cuda":
        operator = SO2CUDAOperator(irreps_in, irreps_out, mmax, weights, geometry)
    elif implementation == "activation":
        operator = ActivationOperator(irreps_in, irreps_out, mmax, weights, geometry)
    elif implementation == "explicit_gemm":
        operator = ExplicitGEMMOperator(irreps_in, irreps_out, mmax, weights, geometry)
    elif implementation == "cueq":
        operator = CueqOperator(irreps_in, irreps_out, mmax, weights, geometry,
                                method="naive", rotation="pytorch", descriptor_name=descriptor_name)
    else:
        operator = Eqv3Operator(irreps_in, irreps_out, mmax, weights,
                                geometry.rotation_matrix, eqv3_root)
    _assert_snapshot(_snapshot(operator, x, cotangent),
                     _snapshot(reference, x.double(), cotangent.double()))


@pytest.mark.parametrize("irreps_in,irreps_out,mmax", _SHAPES + (
    ("2x2e+1x0e+3x1o+1x2e", "1x1o+2x3o+2x0e+1x1o", 1),
))
@pytest.mark.parametrize("descriptor_name", ["escn_tp", "escn_tp_compact"])
def test_cueq_naive_cpu_mapping_gradients_and_equivariance(
        descriptor_name, irreps_in, irreps_out, mmax):
    """Both descriptors work without CUDA, including repeated irreps."""
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
    torch.manual_seed(719)
    weights = canonical_weights(irreps_in, irreps_out, mmax, device="cpu")
    vectors = torch.randn(11, 3)
    x = torch.randn(11, irreps_in.dim)
    cotangent = torch.randn(11, irreps_out.dim)
    lmax = max(irreps_in.lmax, irreps_out.lmax)
    factory = lambda *args: CueqOperator(*args, method="naive", rotation="pytorch",
                                        descriptor_name=descriptor_name)
    operator = factory(irreps_in, irreps_out, mmax, weights, prepare_geometry(vectors, lmax))
    reference = NaiveOperator(irreps_in, irreps_out, mmax,
                              tuple(weight.double() for weight in weights),
                              prepare_geometry(vectors.double(), lmax))
    _assert_snapshot(_snapshot(operator, x, cotangent),
                     _snapshot(reference, x.double(), cotangent.double()))
    _assert_equivariance(factory, irreps_in, irreps_out, weights, vectors, x, mmax)


def test_eqv3_reports_nonuniform_channels_as_unsupported():
    weights = (torch.zeros(3, 3), torch.zeros(2, 1))
    with pytest.raises(ValueError, match="equal channel counts"):
        Eqv3Operator("2x0e+1x1o", "2x0e+1x1o", 1, weights,
                      torch.eye(3)[None], eqv3_root=None)


def test_measure_executes_native_layout_without_canonical_conversions(monkeypatch):
    """Timing must omit the layout wrappers that inflate slice backward cost."""
    class NativeOnly(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(3))
            self.metadata = {"feature_layout": "native [edges, coefficients, channels]"}
            self.calls = 0

        def forward_native(self, x):
            assert x.shape == (2, 4, 3)
            self.calls += 1
            return x @ self.weight.T

        def forward(self, x):
            raise AssertionError("The timed path called the canonical wrapper")

        def input_to_native(self, x):
            raise AssertionError("The timed path converted its input")

        def output_to_native(self, x):
            raise AssertionError("The timed path converted its upstream gradient")

        def output_from_native(self, x):
            raise AssertionError("The timed path converted its output")

    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self):
            pass
        def synchronize(self):
            pass
        def elapsed_time(self, other):
            return 1.

    monkeypatch.setattr(speed, "gpu_snapshot", lambda label: {"label": label, "exclusive": True})
    monkeypatch.setattr(torch.cuda, "Event", Event)
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    for name in ("memory_allocated", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda: 128)
    operator = NativeOnly()
    row = speed.measure(operator, torch.ones(2, 4, 3), torch.ones(2, 4, 3), 1, 2)
    assert operator.calls == 6
    assert row["feature_layout"] == operator.metadata["feature_layout"]
    assert row["memory_scope"]["other_implementations"] == "no live parameters, inputs or geometry"
    assert "CPU only" in row["memory_scope"]["canonical_data"]
    assert row["forward_backward"]["peak_allocated_bytes"] == 128


def test_measurement_setup_releases_non_native_copies(monkeypatch):
    """Setup-only geometry and feature copies must not inflate peak memory."""
    copied = []
    geometry_refs = []

    def to_device(tensor, *args, **kwargs):
        result = tensor.clone()
        copied.append(weakref.ref(result))
        return result

    def geometry(vectors, lmax):
        result = SimpleNamespace(vectors=vectors)
        geometry_refs.append(weakref.ref(vectors))
        return result

    class NativeOnly(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.weight = torch.nn.Parameter(weights[0].clone())

        def input_to_native(self, x):
            return x.clone().reshape(len(x), 4, 3)

        def output_to_native(self, x):
            return x.clone().reshape(len(x), 4, 3)

    monkeypatch.setattr(torch.Tensor, "cuda", to_device)
    monkeypatch.setattr(speed, "prepare_geometry", geometry)
    monkeypatch.setattr(speed, "make_operator",
                        lambda name, ii, io, mmax, weights, geometry, *args: NativeOnly(weights))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    irreps = uniform_irreps(1, 3)
    host = ((torch.eye(3),), torch.ones(2, 12), torch.ones(2, 12), torch.ones(2, 3))
    operator, x, cotangent = speed.prepare_measurement("eqv3", irreps, irreps, 1, host, None)
    assert x.shape == cotangent.shape == (2, 4, 3)
    assert len(copied) == 4
    assert all(ref() is None for ref in copied + geometry_refs)
    assert operator.weight.shape == (3, 3)


@pytest.mark.parametrize("implementation,descriptor", [
    ("naive", "escn_tp"), ("cueq", "escn_tp"), ("cueq", "escn_tp_compact"),
])
def test_geometry_cache_cpu_setup_preserves_exact_outputs_and_gradients(
        monkeypatch, implementation, descriptor):
    """CPU adapters verify the copy/setup contract independently of CUDA."""
    if implementation == "cueq":
        pytest.importorskip("cuequivariance_torch")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor.clone())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    torch.manual_seed(42)
    ii, io = o3.Irreps("3x0e+2x1o+1x2e"), o3.Irreps("2x0e+1x1o+3x2e")
    host = (canonical_weights(ii, io, 2, device="cpu"), torch.randn(7, ii.dim),
            torch.randn(7, io.dim), torch.randn(7, 3))
    cached = speed.prepare_geometry_cache(host[-1], 2)
    snapshots = []
    for geometry in (None, cached):
        op, x, cotangent = speed.prepare_measurement(
            implementation, ii, io, 2, host, None, descriptor_name=descriptor,
            geometry_cpu=geometry)
        snapshots.append(speed.snapshot(op, op.input_from_native(x), op.output_from_native(cotangent)))
    for key in ("output", "input_gradient"):
        assert torch.equal(snapshots[0][key], snapshots[1][key])
    for first, second in zip(snapshots[0]["weight_gradients"], snapshots[1]["weight_gradients"]):
        assert torch.equal(first, second)


@pytest.mark.parametrize("implementation,rotation,fields", [
    ("naive", "pytorch", {"blocks"}), ("cueq", "pytorch", {"blocks"}),
    ("cueq", "cueq", {"alpha", "beta"}),
    ("so2cuda", "pytorch", {"vectors", "blocks"}),
    ("eqv3", "pytorch", {"rotation_matrix"}),
    ("eqv3+compile", "pytorch", {"rotation_matrix"}),
])
def test_geometry_cache_transfers_only_required_fields_and_releases_temporaries(
        monkeypatch, implementation, rotation, fields):
    """A CPU cache must not keep candidate GPU transfer tensors alive."""
    transfers, prepared = [], []

    def transfer(tensor):
        result = tensor.clone()
        transfers.append(weakref.ref(result))
        return result

    def geometry(vectors, lmax):
        values = Geometry(vectors, torch.ones(2), torch.ones(2), torch.eye(3)[None].repeat(2, 1, 1),
                          (torch.ones(2, 1, 1), torch.ones(2, 3, 3)))
        prepared.extend(weakref.ref(value) for value in
                        (values.vectors, values.alpha, values.beta, values.rotation_matrix, *values.blocks))
        return values

    class NativeOnly(torch.nn.Module):
        def __init__(self, weights):
            super().__init__()
            self.weight = torch.nn.Parameter(weights[0].clone())

        def input_to_native(self, x):
            return x.clone()

        output_to_native = input_to_native

    def operator(name, ii, io, mmax, weights, geometry, *args):
        assert {name for name in ("vectors", "alpha", "beta", "rotation_matrix", "blocks")
                if getattr(geometry, name) is not None and
                (name != "blocks" or geometry.blocks)} == fields
        return NativeOnly(weights)

    monkeypatch.setattr(torch.Tensor, "cuda", transfer)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(speed, "prepare_geometry", geometry)
    monkeypatch.setattr(speed, "make_operator", operator)
    ii = uniform_irreps(1, 1)
    host = ((torch.eye(2),), torch.ones(2, 4), torch.ones(2, 4), torch.ones(2, 3))
    cached = speed.prepare_geometry_cache(host[-1], 1)
    assert all(ref() is None for ref in prepared + transfers)
    assert all(tensor.device.type == "cpu" for tensor in
               (cached.vectors, cached.alpha, cached.beta, cached.rotation_matrix, *cached.blocks))
    speed.prepare_measurement(implementation, ii, ii, 1, host, None,
                              rotation=rotation, geometry_cpu=cached)
    assert all(ref() is None for ref in transfers)


@pytest.mark.parametrize("nonuniform", [False, True])
def test_geometry_cache_cuda_setup_preserves_exact_outputs_and_gradients(nonuniform):
    """The cache retains GPU rounding and each native implementation's math."""
    if not torch.cuda.is_available():
        pytest.skip("GPU-computed geometry cache requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(42)
    if nonuniform:
        ii = o3.Irreps(speed.NONUNIFORM_CASES[-1][2])
        io = o3.Irreps(speed.NONUNIFORM_CASES[-1][3])
        mmax = 6
    else:
        ii = io = uniform_irreps(2, 3)
        mmax = 2
    host = (canonical_weights(ii, io, mmax, device="cpu"), torch.randn(7, ii.dim),
            torch.randn(7, io.dim), torch.randn(7, 3))
    cached = speed.prepare_geometry_cache(host[-1], max(ii.lmax, io.lmax))
    assert all(tensor.device.type == "cpu" for tensor in
               (cached.vectors, cached.alpha, cached.beta, cached.rotation_matrix, *cached.blocks))
    implementations = [("naive", "pytorch"), ("so2cuda", "pytorch")]
    if importlib.util.find_spec("cuequivariance_torch"):
        implementations.extend(("cueq", rotation) for rotation in ("pytorch", "cueq"))
    root = os.environ.get("SO2CUDA_EQV3_ROOT")
    if root and not nonuniform:
        implementations.append(("eqv3", "pytorch"))
    for implementation, rotation in implementations:
        snapshots = []
        for geometry in (None, cached):
            op, x, cotangent = speed.prepare_measurement(
                implementation, ii, io, mmax, host, root, rotation=rotation,
                descriptor_name="escn_tp_compact", geometry_cpu=geometry)
            snapshots.append(speed.snapshot(op, op.input_from_native(x), op.output_from_native(cotangent)))
            del op, x, cotangent
        for key in ("output", "input_gradient"):
            assert torch.equal(snapshots[0][key], snapshots[1][key]), (implementation, rotation, key)
        for index, (first, second) in enumerate(zip(snapshots[0]["weight_gradients"],
                                                  snapshots[1]["weight_gradients"])):
            assert torch.equal(first, second), (implementation, rotation, "weight", index)


def test_candidate_keyerror_is_recorded_and_later_candidates_run(monkeypatch, tmp_path):
    """A third-party method bug must not discard other valid candidates."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(speed, "source_identity", lambda: {})
    caches = []
    def cache(*args):
        value = object()
        caches.append(value)
        return value
    monkeypatch.setattr(speed, "prepare_geometry_cache", cache)
    visited = []
    def check(*args, **kwargs):
        method = kwargs.get("method", "naive")
        visited.append((kwargs.get("descriptor_name", "escn_tp"), method))
        if method == "indexed_linear":
            raise KeyError(0)
        return {"passed": True, "implementations": {"cueq": {"status": "passed"}}}
    monkeypatch.setattr(speed, "equivalence_case", check)
    def setup(*args, **kwargs):
        assert kwargs["geometry_cpu"] is caches[0]
        return SimpleNamespace(metadata={}), None, None
    monkeypatch.setattr(speed, "prepare_measurement", setup)
    monkeypatch.setattr(speed, "measure", lambda *a, **k: {"forward_backward": {"median_ms": 1.}})
    args = SimpleNamespace(impl="cueq", include_compile=False, lmax=1, channels=1,
        irreps_in=None, irreps_out=None, mmax=None, edges=2, check_only=False,
        eqv3_root=None, json=tmp_path / "result.json", warmup=5, iterations=20)
    result = speed.run(args)
    rows = result["cueq_alternatives"]
    assert result["status"] == "completed" and len(rows) == 16
    failures = [row for row in rows if row["status"] == "unavailable"]
    assert len(failures) == 4 and all(row["reason"] == "KeyError: 0" for row in failures)
    assert ("escn_tp_compact", "naive") in visited
    assert result["implementations"]["cueq"]["status"] == "passed"
    assert len(caches) == 1


def test_so2cuda_candidate_control_restores_environment(monkeypatch):
    from operator_baselines import so2cuda_candidate_environment
    monkeypatch.setenv("SO2_CUDA_FORWARD_MODE", "scalar")
    monkeypatch.delenv("DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE", raising=False)
    with pytest.raises(RuntimeError):
        with so2cuda_candidate_environment("dense_pairs_grouped"):
            assert os.environ["SO2_CUDA_FORWARD_MODE"] == "indexed_sandwich_multi_grouped"
            assert os.environ["DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE"] == "indexed_sandwich_multi_grouped"
            raise RuntimeError("candidate unavailable")
    assert os.environ["SO2_CUDA_FORWARD_MODE"] == "scalar"
    assert "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE" not in os.environ


def test_requested_suite_reuses_shared_uniform_rows_and_covers_all_groups():
    cases = speed.benchmark_cases("all")
    assert len(cases) == 26
    counts = {group: sum(group in case["groups"] for case in cases) for group in ("A1", "A2", "A3", "A4")}
    assert counts == {"A1": 9, "A2": 6, "A3": 1, "A4": 12}
    assert len(speed.benchmark_cases("uniform")) == 14
    assert len(speed.benchmark_cases("nonuniform")) == 12
    shared = [case for case in cases if case["groups"] == ["A1", "A2"]]
    assert {case["config"]["channels"] for case in shared} == {32, 128}
    assert all(case["config"]["edges"] == 50000 for case in shared)
    assert all(case["config"]["mmax"] == 2 for case in cases if "A3" in case["groups"])


@pytest.mark.parametrize("case_name,label,irreps_in,irreps_out", speed.NONUNIFORM_CASES)
def test_requested_nonuniform_shape_small_cuda_equivalence(case_name, label, irreps_in, irreps_out):
    if not torch.cuda.is_available():
        pytest.skip("Public dense operator equivalence requires CUDA")
    # Preserve the degree/channel pattern with smaller multiplicities so this
    # regression exercises every new shape without allocating benchmark sizes.
    reduce = lambda irreps: o3.Irreps([(max(1, mul // 16), ir) for mul, ir in o3.Irreps(irreps)])
    ii, io = reduce(irreps_in), reduce(irreps_out or irreps_in)
    implementations = ["naive", "so2cuda", "eqv3"]
    if importlib.util.find_spec("cuequivariance_torch") is not None:
        implementations.append("cueq")
    result = speed.equivalence_case(ii, io, min(ii.lmax, io.lmax), edges=9,
                                   eqv3_root=os.environ.get("SO2CUDA_EQV3_ROOT"),
                                   implementations=implementations)
    assert result["passed"], result
    assert result["implementations"]["so2cuda"]["metadata"]["api"].endswith("true_dense_pairs")
    if os.environ.get("SO2CUDA_EQV3_ROOT"):
        assert result["implementations"]["eqv3"]["status"] == "N/A"


def test_default_so2cuda_measurement_uses_public_default_api(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(speed, "source_identity", lambda: {})
    monkeypatch.setattr(speed, "prepare_geometry_cache", lambda *args: None)
    monkeypatch.setattr(speed, "equivalence_case", lambda *a, **k:
                        {"passed": True, "implementations": {"so2cuda": {"status": "passed"}}})
    calls = []
    def setup(*args, **kwargs):
        calls.append(kwargs.get("so2cuda_candidate", "true_dense_pairs"))
        assert "SO2_CUDA_FORWARD_MODE" not in os.environ
        assert "DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE" not in os.environ
        return SimpleNamespace(metadata={"api": "true_dense_pairs"}), None, None
    monkeypatch.setattr(speed, "prepare_measurement", setup)
    monkeypatch.setattr(speed, "measure", lambda *a, **k:
                        {"forward": {"median_ms": 1.}, "forward_backward": {"median_ms": 2.}})
    args = SimpleNamespace(impl="so2cuda", include_compile=False, lmax=1, channels=1,
        irreps_in=None, irreps_out=None, mmax=None, edges=2, check_only=False,
        eqv3_root=None, json=None, warmup=5, iterations=20)
    result = speed.run(args)
    assert calls == ["true_dense_pairs"]
    assert result["implementations"]["so2cuda"]["status"] == "passed"
    assert "so2cuda_alternatives" not in result


def test_small_compiled_equiformerv3_executes_graphs_and_matches_gradients():
    if not torch.cuda.is_available():
        pytest.skip("Compiled operator timing requires CUDA")
    root = os.environ.get("SO2CUDA_EQV3_ROOT")
    if not root:
        pytest.skip("Set SO2CUDA_EQV3_ROOT to the pinned original checkout")
    args = SimpleNamespace(impl="eqv3,eqv3+compile", include_compile=False,
        lmax=1, channels=2, irreps_in=None, irreps_out=None, mmax=None,
        edges=16, check_only=False, eqv3_root=root, json=None, warmup=5, iterations=20,
        allow_shared_gpu=True)
    result = speed.run(args)
    compiled = result["implementations"]["eqv3+compile"]
    assert compiled["status"] == "passed"
    assert compiled["metadata"]["compile_cache_reset"]
    assert compiled["compile_execution"]["executed"]
    assert compiled["compile_execution"]["counter_delta_before_timing"]["unique_graphs"] > 0
    assert compiled["equivalence_vs_eager"]["passed"]


def test_shared_gpu_smoke_is_bounded_before_cuda_execution():
    with pytest.raises(ValueError, match="at most 512 edges"):
        speed.run(SimpleNamespace(allow_shared_gpu=True, edges=513))
    with pytest.raises(ValueError, match="at most 512 edges"):
        speed.measure(SimpleNamespace(), torch.empty(513, 1), None, 5, 20, allow_shared=True)


def test_shared_gpu_snapshot_preserves_strict_default_and_actual_processes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-test"))
    xml = ("<nvidia_smi_log><gpu><processes>"
           f"<process_info><pid>{os.getpid()}</pid><type>C</type><used_memory>128 MiB</used_memory></process_info>"
           "<process_info><pid>999999</pid><type>G</type><used_memory>4 MiB</used_memory></process_info>"
           "</processes></gpu></nvidia_smi_log>")
    monkeypatch.setattr(speed.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=xml))
    with pytest.raises(speed.GPUExclusivityError):
        speed.gpu_snapshot("before")
    proof = speed.gpu_snapshot("before", allow_shared=True)
    assert proof["exclusive"] is False and proof["allow_shared_gpu"] is True
    assert [row["type"] for row in proof["processes"]] == ["C", "G"]
    assert "functional smoke" in proof["reason"]
