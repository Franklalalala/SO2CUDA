"""Uniform dispatch, output and both gradients against the independent FP64 reference."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from operator_baselines import uniform_irreps
from so2_operator_speed_test import equivalence_case

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("lmax", [2, 4, 6])
@pytest.mark.parametrize("channels", [32, 128])
@pytest.mark.parametrize("truncated", [False, True])
def test_uniform_fp64(monkeypatch, lmax, channels, truncated):
    from so2_cuda_ops import _sandwich
    ext = _sandwich._ext()
    original = ext.uniform_transform_fp32
    calls = []

    def traced(*args):
        calls.append(bool(args[-1]))
        return original(*args)

    monkeypatch.setattr(ext, "uniform_transform_fp32", traced)
    if channels == 32 and lmax <= 2:
        # Exercise the specialized kernel against FP64 even for layouts whose
        # default dispatch keeps the faster general training path.
        original_plan = _sandwich.sandwich_plan

        def forced_plan(*args, **kwargs):
            plan = original_plan(*args, **kwargs)
            assert plan.inp.uniform == plan.out.uniform == (channels, lmax)
            plan.uniform = True
            return plan

        monkeypatch.setattr(_sandwich, "sandwich_plan", forced_plan)
    irreps = uniform_irreps(lmax, channels)
    result = equivalence_case(irreps, irreps, min(2, lmax - 1) if truncated else lmax,
                              edges=17, implementations=("so2cuda",), equivariance=False,
                              so2cuda_candidate="true_dense_pairs")
    assert result["passed"], result["comparisons"]
    assert False in calls and True in calls, "uniform rotation and gather must execute"


@pytest.mark.parametrize("with_m0", [False, True])
def test_uniform_identity_and_empty(with_m0):
    from so2_cuda_ops import _sandwich
    from so2_cuda_ops.deeptb import prepare_layout, prepare_wigner
    entries = ((0, 64, 0), (1, 64, 64), (2, 64, 256))
    layout = prepare_layout(entries, entries, m_max=1, l_max=2, out_dim=576, device="cuda")
    for n in (0, 7):
        x = torch.randn(n, 576, device="cuda")
        wigner = prepare_wigner(x, None, l_max=2, rotate=False)
        plan = _sandwich.sandwich_plan(layout, 576, x.device, with_m0=with_m0)
        assert plan.uniform
        packed = _sandwich._rotate(x, plan.inp, plan, wigner, False)
        actual = _sandwich._gather(packed, n, plan.out, plan, wigner, False)
        plan.uniform = False
        reference = _sandwich._gather(
            _sandwich._rotate(x, plan.inp, plan, wigner, False), n, plan.out, plan, wigner, False)
        plan.uniform = True
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize("mmax", [0, 1, 2])
def test_small_uniform_uses_general_training_path(monkeypatch, mmax):
    from so2_cuda_ops import _sandwich

    def unexpected(*args):
        pytest.fail("small uniform layouts must keep the general training path")

    monkeypatch.setattr(_sandwich._ext(), "uniform_transform_fp32", unexpected)
    irreps = uniform_irreps(2, 32)
    result = equivalence_case(irreps, irreps, mmax, edges=17,
                              implementations=("so2cuda",), equivariance=False,
                              so2cuda_candidate="true_dense_pairs")
    assert result["passed"], result["comparisons"]


def test_nonuniform_keeps_general_plan():
    from so2_cuda_ops import _sandwich
    from so2_cuda_ops.deeptb import prepare_layout
    entries = ((0, 64, 0), (1, 32, 64), (2, 32, 160))
    layout = prepare_layout(entries, entries, m_max=2, l_max=2, out_dim=320, device="cuda")
    assert not _sandwich.sandwich_plan(layout, 320, "cuda", with_m0=True).uniform


def test_different_uniform_sides_keep_general_plan():
    from so2_cuda_ops import _sandwich
    from so2_cuda_ops.deeptb import prepare_layout
    entries = ((0, 32, 0), (1, 32, 32), (2, 32, 128))
    other = ((0, 64, 0), (1, 64, 64), (2, 64, 256))
    layout = prepare_layout(entries, other, m_max=2, l_max=2, out_dim=576, device="cuda")
    assert not _sandwich.sandwich_plan(layout, 288, "cuda", with_m0=True).uniform


def test_uniform_only_m0_fp64():
    irreps = uniform_irreps(2, 32)
    result = equivalence_case(irreps, irreps, 0, edges=17, implementations=("so2cuda",),
                              equivariance=False, so2cuda_candidate="true_dense_pairs")
    assert result["passed"], result["comparisons"]


@pytest.mark.parametrize("missing", ["interior", "trailing"])
@pytest.mark.parametrize("with_m0", [False, True])
def test_missing_positive_m_blocks_use_general_path(monkeypatch, missing, with_m0):
    from so2_cuda_ops import _sandwich
    from so2_cuda_ops.deeptb import LinearWeights, prepare_layout, prepare_wigner, true_dense_pairs

    def unexpected(*args):
        pytest.fail("missing positive m blocks must use the general path")

    monkeypatch.setattr(_sandwich._ext(), "uniform_transform_fp32", unexpected)
    channels, lmax = 64, 2
    dim = channels * (lmax + 1) ** 2
    entries = tuple((l, channels, channels * l * l) for l in range(lmax + 1))
    mmax = 2 if missing == "interior" else 4
    layout = prepare_layout(entries, entries, m_max=mmax, l_max=mmax, out_dim=dim,
                            device="cuda", rotate_in=False, rotate_out=False)
    if missing == "interior":
        # Construct the descriptor before its first use, with no m=1 operator.
        maps = list(layout.maps)
        empty = torch.empty(0, dtype=torch.long, device="cuda")
        maps[1] = (empty, empty, empty, empty, maps[1][4])
        layout.maps = tuple(maps)
    x = (torch.arange(dim, device="cuda", dtype=torch.float32)[None] + 1).requires_grad_()
    wigner = prepare_wigner(x, None, l_max=mmax, rotate=False)
    plan = _sandwich.sandwich_plan(layout, dim, x.device, with_m0=with_m0)
    assert not plan.uniform
    assert plan.inp.uniform is None and plan.out.uniform is None
    weights = []
    for m, mapping in enumerate(layout.maps):
        width = mapping[0].numel()
        real = torch.eye(width, device="cuda")
        weight = real if m == 0 else torch.cat((real, torch.zeros_like(real)))
        weights.append(LinearWeights(weight, routed=False))
    actual = true_dense_pairs(x, layout, wigner, tuple(weights), include_m0=with_m0)[0]
    # Identity real weights retain exactly the represented spherical components.
    mask = torch.zeros_like(x)
    for l in range(lmax + 1):
        base = channels * l * l
        for j in range(2 * l + 1):
            m = abs(j - l)
            if (m > 0 or with_m0) and (missing != "interior" or m != 1):
                mask[:, base + j:base + channels * (2 * l + 1):2 * l + 1] = 1
    upstream = torch.linspace(-1, 1, dim, device="cuda")[None]
    dx, = torch.autograd.grad(actual, x, upstream)
    torch.testing.assert_close(actual, x * mask, atol=0, rtol=0)
    torch.testing.assert_close(dx, upstream * mask, atol=0, rtol=0)


def test_equal_dimensions_with_different_uniform_sides_fp64(monkeypatch):
    from so2_cuda_ops import _sandwich

    def unexpected(*args):
        pytest.fail("equal total dimensions do not imply equal uniform layouts")

    monkeypatch.setattr(_sandwich._ext(), "uniform_transform_fp32", unexpected)
    # Both sides have 1152 features, but their channel counts and degrees differ.
    result = equivalence_case(uniform_irreps(2, 128), uniform_irreps(5, 32), 2,
                              edges=17, implementations=("so2cuda",), equivariance=False,
                              so2cuda_candidate="true_dense_pairs")
    assert result["passed"], result["comparisons"]
