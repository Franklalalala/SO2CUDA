"""The model cueq route preserves shared Wigner and canonical gradients."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
pytest.importorskip("dptb")
pytest.importorskip("cuequivariance")
pytest.importorskip("cuequivariance_torch")

from dptb.nn.pdq_moe import MOLEGlobals
from dptb.nn.tensor_product_moe_v3 import SO2_Linear
import cueq_baseline
from cueq_baseline import CueqSO2Linear, cueq_execution_metadata
from naive_baseline import NaiveSO2Linear


@pytest.mark.parametrize("descriptor", ["escn_tp", "escn_tp_compact"])
@pytest.mark.parametrize("front", [False, True])
def test_split_rotation_radial_gradients_and_cache(monkeypatch, descriptor, front):
    wider, narrower = "2x0e+3x1o+1x0e+1x1o+2x2e", "3x0e+2x1o+2x2e+1x1o"
    irreps_in, irreps_out = (narrower, wider) if front else (wider, narrower)
    source = SO2_Linear(irreps_in, irreps_out,
                       radial_emb=True, latent_dim=4, radial_channels=[4],
                       num_experts=1, num_shared_experts=0)
    assert source.front is front
    generator = torch.Generator().manual_seed(314)
    x = torch.randn(7, source.irreps_in.dim, generator=generator).requires_grad_()
    vectors = torch.randn(7, 3, generator=generator)
    latents = torch.randn(7, 4, generator=generator).requires_grad_()
    routes = MOLEGlobals(coefficients=torch.ones(1, 1), graph_index=torch.zeros(7, dtype=torch.long))
    targets = (x, latents) + tuple(source.parameters())
    expected, _ = NaiveSO2Linear(source)(x, vectors, routes, latents)
    upstream = torch.randn(expected.shape, generator=generator)
    reference_gradients = torch.autograd.grad(expected, targets, upstream)

    operator = CueqSO2Linear(source, method="naive", descriptor=descriptor)
    actual, cache = operator(x, vectors, routes, latents)
    actual_gradients = torch.autograd.grad(actual, targets, upstream)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=5e-5)
    for reference, value in zip(reference_gradients, actual_gradients):
        torch.testing.assert_close(value, reference, atol=5e-6, rtol=5e-5)
    assert hasattr(cache, "blocks")
    assert {name: id(p) for name, p in operator.named_parameters()} == {
        name: id(p) for name, p in source.named_parameters()}

    def unexpected_build(*args, **kwargs):
        pytest.fail("A supplied model Wigner cache must not be rebuilt")

    monkeypatch.setattr(cueq_baseline, "_make_wigner_rotation", unexpected_build)
    cached, reused = operator(x, vectors, routes, latents, cache)
    assert reused is cache
    torch.testing.assert_close(cached, actual, atol=0, rtol=0)
    metadata = cueq_execution_metadata(operator)[""]
    assert metadata["wigner_geometry_builds"] == 1
    assert metadata["wigner_cache_hits"] == 1
    assert metadata["forward_calls"] == 2
    assert metadata["wigner_shared_per_forward"]
    assert metadata["wigner_layout"] == "compact_blocks"
