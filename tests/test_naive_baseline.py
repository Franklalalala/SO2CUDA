"""CPU contracts for the optional DeePTB benchmark execution replacement."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("e3nn")
pytest.importorskip("dptb.nn.pdq_moe")

from e3nn.o3 import Irreps
from dptb.nn.pdq_moe import PDQMoE, PDQMoERouting
from dptb.nn.tensor_product import SO2LinearCached, SO2_Linear as DenseSO2
from dptb.nn.tensor_product_moe_v3 import SO2_Linear as ExpertSO2


_path = Path(__file__).resolve().parents[1] / "examples" / "naive_baseline.py"
_spec = importlib.util.spec_from_file_location("naive_baseline_under_test", _path)
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


@contextmanager
def forbid_cuda_backend(monkeypatch):
    """Fail even if a CUDA entry merely declines the CPU input."""
    import dptb.nn.so2_backend as backend
    import dptb.nn.tensor_product as dense
    import so2_cuda_ops.deeptb as public

    def forbidden(*args, **kwargs):
        raise AssertionError("The naive baseline called the optional CUDA backend")

    with monkeypatch.context() as patch:
        for name in ("backend", "activation_forward", "dense_forward", "true_dense_forward",
                     "grouped_gemm", "grouped_gemm_multi"):
            patch.setattr(backend, name, forbidden)
        patch.setattr(dense, "true_dense_forward", forbidden)
        for name in ("activation_forward", "dense_pairs", "prepare_layout", "prepare_wigner"):
            patch.setattr(public, name, forbidden)
        yield


def assert_output_and_gradients(actual, expected, inputs, *, atol=3e-6, rtol=3e-5):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    probe = torch.linspace(-0.7, 1.3, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grad = torch.autograd.grad(actual, inputs, probe, retain_graph=True, allow_unused=True)
    expected_grad = torch.autograd.grad(expected, inputs, probe, allow_unused=True)
    for a, b in zip(actual_grad, expected_grad):
        if a is None or b is None:
            assert a is b, "Execution changed whether a parameter participates in autograd"
        else:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def explicit_mixed_linear(source, x, coefficients, branch="all"):
    """Tiny independent UMA-style per-row weight mixture, including biases."""
    n = x.shape[0]
    weight = x.new_zeros(n, source.out_features, source.in_features)
    bias = x.new_zeros(n, source.out_features)
    if branch != "shared" and source.num_experts:
        for e in range(source.num_experts):
            if source.mole_expert_parameterization == "shared_core":
                w = source.basis_left @ source.core_experts[e] @ source.basis_right.T
                w = source._parity_value("weight_experts", w)
            else:
                w = source.weight_experts[e]
            weight = weight + coefficients[:, e, None, None] * w
            if source.bias_experts is not None:
                bias = bias + coefficients[:, e, None] * source.bias_experts[e]
    if branch != "routed" and source.num_shared_experts:
        weight = weight + source.weight_shared.sum(0)
        if source.bias_shared is not None:
            bias = bias + source.bias_shared.sum(0)
    if x.ndim == 2:
        return torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias
    return torch.bmm(x, weight.transpose(1, 2)) + bias.unsqueeze(1)


@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
@pytest.mark.parametrize("pair_input", [False, True])
@pytest.mark.parametrize("topk", [False, True])
def test_expert_sum_matches_explicit_row_weight_mixture(monkeypatch, parameterization, pair_input, topk):
    torch.manual_seed(103)
    source = PDQMoE(5, 4, num_experts=3, num_shared_experts=2, bias=True,
                   mole_expert_parameterization=parameterization, mole_expert_rank=2)
    naive = baseline.NaiveExpertLinear(source)
    x = torch.randn((6, 2, 5) if pair_input else (6, 5), requires_grad=True)
    logits = torch.randn(6, 3, requires_grad=True)
    if topk:
        indices = torch.tensor([[0, 2], [1, 2], [0, 1], [2, 0], [1, 0], [2, 1]])
        values = logits.gather(1, indices).softmax(1)
        coefficients = torch.zeros_like(logits).scatter_add(1, indices, values)
        route = PDQMoERouting(coefficients=coefficients, topk_indices=indices,
                             topk_values=values, activation_space=True)
    else:
        coefficients = logits.softmax(1)
        route = PDQMoERouting(coefficients=coefficients, activation_space=True)
    expected = explicit_mixed_linear(source, x, coefficients)
    with forbid_cuda_backend(monkeypatch):
        actual = naive(x, route)
        assert_output_and_gradients(actual, expected, (x, logits, *source.parameters()))


@pytest.mark.parametrize("route_kind", ["graph_index", "split_sizes", "broadcast", "default"])
def test_graph_and_default_routing(monkeypatch, route_kind):
    torch.manual_seed(107)
    source = PDQMoE(3, 2, num_experts=3, num_shared_experts=1, bias=False,
                   mole_expert_parameterization="shared_core", mole_expert_rank=2)
    x = torch.randn(5, 2, 3, requires_grad=True)
    logits = torch.randn(3, 3, requires_grad=True)
    coefficients = logits.softmax(1)
    if route_kind == "graph_index":
        index = torch.tensor([2, 0, 1, 2, 0])
        routing = PDQMoERouting(coefficients=coefficients, graph_index=index)
        rows = coefficients[index]
    elif route_kind == "split_sizes":
        routing = PDQMoERouting(coefficients=coefficients, split_sizes=(2, 0, 3))
        rows = coefficients.repeat_interleave(torch.tensor([2, 0, 3]), dim=0)
    elif route_kind == "broadcast":
        routing = PDQMoERouting(coefficients=coefficients[:1])
        rows = coefficients[:1].expand(5, -1)
    else:
        routing = None
        rows = x.new_full((5, 3), 1 / 3)
    expected = explicit_mixed_linear(source, x, rows)
    with forbid_cuda_backend(monkeypatch):
        actual = baseline.NaiveExpertLinear(source)(x, routing)
        assert_output_and_gradients(actual, expected, (x, logits, *source.parameters()))


@pytest.mark.parametrize("topk_scope", ["group", "edge"])
def test_interleaved_groups_mix_group_weights_and_call_linear_once(monkeypatch, topk_scope):
    torch.manual_seed(104)
    source = PDQMoE(3, 2, num_experts=3, num_shared_experts=0, bias=True,
                   mole_linear_mode="indexed_ref", mole_expert_parameterization="shared_core",
                   mole_expert_rank=2)
    x = torch.randn(9, 2, 3, requires_grad=True)
    index = torch.tensor([2, 0, 1, 0, 2, 1, 0, 1, 2])
    logits = torch.randn(3, 3, requires_grad=True)
    indices = torch.tensor([[0, 2], [1, 2], [0, 1]])
    values = logits.gather(1, indices).softmax(1)
    coefficients = torch.zeros_like(logits).scatter_add(1, indices, values)
    routing = PDQMoERouting(coefficients=coefficients, graph_index=index,
                           topk_indices=indices if topk_scope == "group" else indices[index],
                           topk_values=values if topk_scope == "group" else values[index])
    expected = source(x, routing)
    original = baseline.F.linear
    sizes = []

    def recorded(inp, weight, bias=None):
        sizes.append(inp.shape[0])
        return original(inp, weight, bias)

    naive = baseline.NaiveExpertLinear(source)
    assert naive._coefficients(x, routing).shape == (3, 3)
    with monkeypatch.context() as patch:
        patch.setattr(baseline.F, "linear", recorded)
        actual = naive(x, routing)
    assert sizes == [3, 3, 3]
    assert_output_and_gradients(actual, expected, (x, logits, *source.parameters()))


def test_sparse_experts_only_process_selected_edges(monkeypatch):
    torch.manual_seed(105)
    source = PDQMoE(3, 2, num_experts=4, num_shared_experts=1,
                   mole_expert_parameterization="shared_core", mole_expert_rank=2)
    x = torch.randn(4, 2, 3, requires_grad=True)
    indices = torch.tensor([[0, 1], [0, 2], [1, 2], [0, 1]])
    values = torch.randn(4, 2).softmax(1).requires_grad_()
    coefficients = torch.zeros(4, 4).scatter_add(1, indices, values)
    routing = PDQMoERouting(coefficients=coefficients, topk_indices=indices,
                           topk_values=values, activation_space=True)
    expected = explicit_mixed_linear(source, x, coefficients)
    original = baseline.F.linear
    sizes = []

    def recorded(inp, weight, bias=None):
        sizes.append(inp.shape[0])
        return original(inp, weight, bias)

    with monkeypatch.context() as patch:
        patch.setattr(baseline.F, "linear", recorded)
        actual = baseline.NaiveExpertLinear(source)(x, routing)
    assert sizes == [3, 3, 2, 0, 4]
    assert_output_and_gradients(actual, expected, (x, values, *source.parameters()))


@pytest.mark.parametrize("num_experts,num_shared", [(1, 0), (4, 1), (0, 2)])
def test_dense_and_shared_only_expert_counts(monkeypatch, num_experts, num_shared):
    torch.manual_seed(108)
    source = PDQMoE(3, 2, num_experts=num_experts, num_shared_experts=num_shared)
    x = torch.randn(4, 3, requires_grad=True)
    coefficients = x.new_full((4, num_experts), 1 / max(1, num_experts))
    route = PDQMoERouting(coefficients=coefficients, activation_space=True)
    expected = explicit_mixed_linear(source, x, coefficients)
    with forbid_cuda_backend(monkeypatch):
        actual = baseline.NaiveExpertLinear(source)(x, route)
        assert_output_and_gradients(actual, expected, (x, *source.parameters()))


@pytest.mark.parametrize("branch", ["all", "routed", "shared"])
@pytest.mark.parametrize("parameterization", ["full", "shared_core"])
def test_expert_branches_and_parity_masks(monkeypatch, branch, parameterization):
    torch.manual_seed(109)
    source = PDQMoE(3, 2, num_experts=2, num_shared_experts=1,
                   mole_expert_parameterization=parameterization, mole_expert_rank=2)
    source.set_parity_masks(torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool),
                            torch.tensor([True, False]))
    x = torch.randn(4, 3, requires_grad=True)
    logits = torch.randn(4, 2, requires_grad=True)
    coefficients = logits.softmax(1)
    route = PDQMoERouting(coefficients=coefficients, activation_space=True, branch=branch)
    expected = explicit_mixed_linear(source, x, coefficients, branch)
    with forbid_cuda_backend(monkeypatch):
        actual = baseline.NaiveExpertLinear(source)(x, route)
        assert_output_and_gradients(actual, expected, (x, logits, *source.parameters()))


@pytest.mark.parametrize("front", [False, True])
@pytest.mark.parametrize("rotate_in,rotate_out", [(False, False), (True, False),
                                                 (False, True), (True, True)])
@pytest.mark.parametrize("wigner_mode", ["full_dense", "compact_blocks"])
def test_so2_matches_staged_cpu(monkeypatch, front, rotate_in, rotate_out, wigner_mode):
    torch.manual_seed(113)
    narrow = Irreps("2x0e+1x1o+1x2e")
    wide = Irreps("3x0e+2x1o+2x2e")
    irreps_in, irreps_out = (narrow, wide) if front else (wide, narrow)
    source = ExpertSO2(irreps_in, irreps_out, radial_emb=True, latent_dim=4,
                       radial_channels=[5], extra_m0_outsize=1,
                       num_experts=3, num_shared_experts=1, rotate_in=rotate_in,
                       rotate_out=rotate_out, wigner_apply_mode=wigner_mode,
                       mole_linear_mode="split_loop", mole_expert_parameterization="shared_core",
                       mole_expert_rank=2, so2_fusion_mode="staged")
    assert source.front is front
    x = torch.randn(5, irreps_in.dim, requires_grad=True)
    vectors = torch.randn(5, 3, requires_grad=True)
    latents = torch.randn(5, 4, requires_grad=True)
    logits = torch.randn(5, 3, requires_grad=True)
    indices = torch.tensor([[0, 2], [1, 0], [2, 1], [0, 1], [2, 0]])
    values = logits.gather(1, indices).softmax(1)
    coefficients = torch.zeros_like(logits).scatter_add(1, indices, values)
    routing = PDQMoERouting(coefficients=coefficients, topk_indices=indices,
                           topk_values=values, activation_space=True)
    expected, _ = source(x, vectors, routing, latents)
    with forbid_cuda_backend(monkeypatch):
        actual, cache = baseline.NaiveSO2Linear(source)(x, vectors, routing, latents)
        assert cache.shape == (5, 9, 9)
        assert_output_and_gradients(actual, expected,
                                   (x, vectors, latents, logits, *source.parameters()),
                                   atol=8e-6, rtol=8e-5)


@pytest.mark.parametrize("branch", ["all", "routed", "shared"])
def test_interpolation_blocks_and_extra_scalar_outputs(monkeypatch, branch):
    torch.manual_seed(127)
    irreps = Irreps("2x0e+2x1o+1x2e")
    source = ExpertSO2(irreps, irreps, extra_m0_outsize=2, use_interpolation=True,
                       num_experts=2, num_shared_experts=1, rotate_in=True, rotate_out=True,
                       mole_linear_mode="split_loop", so2_fusion_mode="staged")
    x = torch.randn(4, irreps.dim, requires_grad=True)
    vectors = torch.randn(4, 3)
    coefficients = torch.randn(4, 2).softmax(1).requires_grad_()
    indices = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
    routing = PDQMoERouting(coefficients=coefficients, topk_indices=indices,
                           topk_values=coefficients.gather(1, indices),
                           activation_space=True, branch=branch)
    expected, _ = source(x, vectors, routing)
    with forbid_cuda_backend(monkeypatch):
        actual, _ = baseline.NaiveSO2Linear(source)(x, vectors, routing)
        assert_output_and_gradients(actual, expected, (x, coefficients, *source.parameters()))


@pytest.mark.parametrize("source_kind", ["upstream_interface", "cached_interface"])
def test_dense_positional_and_keyword_latents(monkeypatch, source_kind):
    torch.manual_seed(131)
    irreps = Irreps("2x0e+1x1o")
    if source_kind == "upstream_interface":
        source = DenseSO2(irreps, irreps, radial_emb=True, latent_dim=3, radial_channels=[4])
        source.so2_m_linear_mode = "standard"
    else:
        source = SO2LinearCached(irreps, irreps, radial_emb=True, latent_dim=3,
                                radial_channels=[4], so2_m_linear_mode="standard")
    x = torch.randn(4, irreps.dim, requires_grad=True)
    vectors = torch.randn(4, 3)
    latents = torch.randn(4, 3, requires_grad=True)
    expected = source(x, vectors, latents=latents)
    expected_tensor = expected[0] if isinstance(expected, tuple) else expected
    naive = baseline.NaiveSO2Linear(source)
    with forbid_cuda_backend(monkeypatch):
        positional = naive(x, vectors, latents)
        keyword = naive(x, vectors, latents=latents)
        for actual in (positional, keyword):
            assert isinstance(actual, tuple) == isinstance(expected, tuple)
            tensor = actual[0] if isinstance(actual, tuple) else actual
            torch.testing.assert_close(tensor, expected_tensor, atol=3e-6, rtol=3e-5)
        actual = keyword[0] if isinstance(keyword, tuple) else keyword
        assert_output_and_gradients(actual, expected_tensor, (x, latents, *source.parameters()))


def test_execution_swap_preserves_parameters_names_buffers_and_restore(monkeypatch):
    torch.manual_seed(137)
    irreps = Irreps("2x0e+1x1o")
    model = torch.nn.Module()
    model.embedding = torch.nn.Module()
    model.embedding.so2_expert_mixing_mode = "pre_activation"
    model.embedding.tp = ExpertSO2(irreps, irreps, num_experts=2, num_shared_experts=1,
                                  mole_expert_parameterization="shared_core", mole_expert_rank=2,
                                  so2_parity="enforce", so2_fusion_mode="staged")
    model.extra = PDQMoE(3, 2, num_experts=1, num_shared_experts=0)
    model.eval()
    original_tp, original_extra = model.embedding.tp, model.extra
    parameters = dict(model.named_parameters())
    state = {name: value.clone() for name, value in model.state_dict().items()}
    buffers = dict(model.named_buffers())
    handle = baseline.install_naive_baseline(model)
    assert isinstance(model.embedding.tp, baseline.NaiveSO2Linear)
    assert isinstance(model.extra, baseline.NaiveExpertLinear)
    assert not model.embedding.tp.training
    assert dict(model.named_parameters()).keys() == parameters.keys()
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())
    assert dict(model.named_buffers()).keys() == buffers.keys()
    assert all(dict(model.named_buffers())[name] is value for name, value in buffers.items())
    assert model.state_dict().keys() == state.keys()
    model.load_state_dict(state, strict=True)
    with forbid_cuda_backend(monkeypatch):
        route = PDQMoERouting(coefficients=torch.ones(3, 2) / 2, activation_space=True)
        model.embedding.tp(torch.randn(3, irreps.dim), torch.randn(3, 3), route)
        model.extra(torch.randn(3, 3), None)
    model.train()
    handle.restore()
    assert model.embedding.tp is original_tp
    assert model.extra is original_extra
    assert original_tp.training and original_extra.training
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())
    handle.restore()


def test_wigner_is_rebuilt_per_call_and_supplied_cache_is_ignored(monkeypatch):
    torch.manual_seed(139)
    irreps = Irreps("2x0e+1x1o")
    source = ExpertSO2(irreps, irreps, num_experts=1, num_shared_experts=0,
                       wigner_apply_mode="compact_blocks", so2_fusion_mode="staged")
    naive = baseline.NaiveSO2Linear(source)
    x = torch.randn(3, irreps.dim)
    vectors = torch.randn(3, 3)
    routing = PDQMoERouting(coefficients=torch.ones(1, 1), split_sizes=(3,))
    original = baseline.batch_wigner_D
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(baseline, "batch_wigner_D", counted)
    with forbid_cuda_backend(monkeypatch):
        first, first_cache = naive(x, vectors, routing)
        second, second_cache = naive(x, vectors, routing, wigner_D_all=torch.full((3, 4, 4), 999.0))
    assert calls == [1, 1]
    assert first_cache.data_ptr() != second_cache.data_ptr()
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    torch.testing.assert_close(first_cache, second_cache, atol=0, rtol=0)
