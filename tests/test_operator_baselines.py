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
    CueqOperator, NaiveOperator, SO2CUDAOperator, ExplicitGEMMOperator,
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
    ("so2cuda", None), ("explicit_gemm", None), ("cueq", "escn_tp"),
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


def test_candidate_keyerror_is_recorded_and_later_candidates_run(monkeypatch, tmp_path):
    """A third-party method bug must not discard other valid candidates."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "test device")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(speed, "source_identity", lambda: {})
    visited = []
    def check(*args, **kwargs):
        method = kwargs.get("method", "naive")
        visited.append((kwargs.get("descriptor_name", "escn_tp"), method))
        if method == "indexed_linear":
            raise KeyError(0)
        return {"passed": True, "implementations": {"cueq": {"status": "passed"}}}
    monkeypatch.setattr(speed, "equivalence_case", check)
    monkeypatch.setattr(speed, "prepare_measurement", lambda *a, **k:
                        (SimpleNamespace(metadata={}), None, None))
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
