"""Numerical tests of the tensor-only integration boundary."""
import pytest

torch = pytest.importorskip('torch')


def test_cpu_optional_backend_declines_without_model_dependency():
    from so2_cuda_ops.deeptb import (
        ActivationRouting, DenseRouting, LinearWeights, activation_forward,
        dense_pairs, prepare_layout, prepare_wigner,
    )
    x = torch.randn(7, 5)
    layout = prepare_layout(((0, 2, 0), (1, 1, 2)), ((0, 2, 0), (1, 1, 2)),
                            m_max=1, l_max=1, out_dim=5, device=x.device,
                            rotate_in=False, rotate_out=False)
    wigner = prepare_wigner(x, None, l_max=1, rotate=False)
    weights = (torch.randn(1, 3, 3), torch.randn(1, 2, 1))
    route = DenseRouting(torch.zeros(7, dtype=torch.long), torch.tensor([0, 14]))
    assert dense_pairs(x, layout, wigner, weights, None, route) is None
    idx = torch.zeros(7, 1, dtype=torch.long)
    activation_route = ActivationRouting(idx, torch.ones(7, 1), ())
    assert activation_forward(x, layout, wigner, tuple(LinearWeights(w) for w in weights),
                              None, activation_route) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason='SO2 integration correctness requires CUDA')
def test_activation_matches_independent_torch_after_inference_warmup():
    from so2_cuda_ops.deeptb import (
        ActivationRouting, LinearWeights, activation_forward, prepare_layout, prepare_wigner,
    )
    torch.manual_seed(21)
    n = 13
    x = torch.randn(n, 5, device='cuda', requires_grad=True)
    w0 = torch.randn(2, 3, 3, device='cuda', requires_grad=True)
    w1 = torch.randn(2, 2, 1, device='cuda', requires_grad=True)
    val = torch.rand(n, 2, device='cuda', requires_grad=True)
    idx = torch.stack((torch.arange(n, device='cuda') % 2,
                       1 - torch.arange(n, device='cuda') % 2), dim=1)
    slots = []
    for j in range(2):
        order = torch.argsort(idx[:, j], stable=True)
        inverse = torch.argsort(order)
        counts = torch.bincount(idx[:, j], minlength=2).cpu()
        ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        slots.append((order, inverse, ptr, idx[:, j][order]))
    routing = ActivationRouting(idx, val, tuple(slots))
    params = (LinearWeights(w0), LinearWeights(w1))
    with torch.inference_mode():
        layout = prepare_layout(((0, 2, 0), (1, 1, 2)), ((0, 2, 0), (1, 1, 2)),
                                m_max=1, l_max=1, out_dim=5, device=x.device,
                                rotate_in=False, rotate_out=False)
        warm_wigner = prepare_wigner(x, None, l_max=1, rotate=False)
        activation_forward(x, layout, warm_wigner, params, None, routing)
    wigner = prepare_wigner(x, None, l_max=1, rotate=False)
    actual = activation_forward(x, layout, wigner, params, None, routing)
    m0 = x[:, [0, 1, 3]]
    pair = x[:, [2, 4]].unsqueeze(-1)
    expected0, expected_pair = 0, 0
    for j in range(2):
        out0 = torch.bmm(w0[idx[:, j]], m0.unsqueeze(-1)).squeeze(-1)
        raw = torch.matmul(pair, w1[idx[:, j]].transpose(1, 2))
        out_pair = torch.stack((raw[:, 0, 0]-raw[:, 1, 1],
                                raw[:, 1, 0]+raw[:, 0, 1]), dim=1)
        expected0 = expected0 + val[:, j, None]*out0
        expected_pair = expected_pair + val[:, j, None]*out_pair
    expected = torch.stack((expected0[:, 0], expected0[:, 1], expected_pair[:, 0],
                            expected0[:, 2], expected_pair[:, 1]), dim=1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    grad = torch.randn_like(actual)
    args = (x, w0, w1, val)
    actual_grad = torch.autograd.grad(actual, args, grad, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, args, grad)
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
