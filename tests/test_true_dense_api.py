"""Tensor-only true-dense pair contributions against an independent reference."""
import pytest
import torch
import torch.nn.functional as F

from so2_cuda_ops.deeptb import LinearWeights, prepare_layout, prepare_wigner, true_dense_pairs


def test_true_dense_declines_cpu_without_native_execution():
    x = torch.ones(3, 4)
    layout = prepare_layout(((0, 1, 0), (1, 1, 1)), ((0, 1, 0), (1, 1, 1)),
                            m_max=1, l_max=1, out_dim=4, device="cpu")
    wigner = prepare_wigner(x, None, l_max=1, rotate=False)
    linears = (LinearWeights(torch.ones(2, 2), routed=False), LinearWeights(torch.ones(2, 1), routed=False))
    assert true_dense_pairs(x, layout, wigner, linears) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("front", [True, False])
@pytest.mark.parametrize("radial", [True, False])
def test_true_dense_pair_forward_and_backward(front, radial):
    small, large = ((0, 1, 0), (1, 2, 1), (2, 1, 7)), ((0, 2, 0), (1, 1, 2), (2, 2, 5))
    inputs, outputs = (small, large) if front else (large, small)
    dim_in, dim_out = (12, 15) if front else (15, 12)
    layout = prepare_layout(inputs, outputs, m_max=2, l_max=2, out_dim=dim_out,
                            device="cuda", rotate_in=False, rotate_out=False, front=front)
    x = torch.randn(7, dim_in, device="cuda", requires_grad=True)
    wigner = prepare_wigner(x, None, l_max=2, rotate=False)
    weights, radials = [], []
    for m in range(3):
        cin = sum(mul for l, mul, _ in inputs if l >= m)
        cout = sum(mul for l, mul, _ in outputs if l >= m)
        weights.append(torch.randn(cout * (2 if m else 1), cin, device="cuda", requires_grad=True))
        radials.append(torch.randn(7, cin if front else cout, device="cuda", requires_grad=True))
    linears = tuple(LinearWeights(w, routed=False) for w in weights)
    parts = true_dense_pairs(x, layout, wigner, linears, tuple(radials) if radial else None)
    assert parts is not None
    actual = sum(parts)
    expected = torch.zeros_like(actual)
    for m in range(1, 3):
        channels = [(l, start + c * (2*l+1)) for l, mul, start in inputs if l >= m for c in range(mul)]
        pair = torch.stack((torch.stack([x[:, base+l-m] for l, base in channels], dim=-1),
                            torch.stack([x[:, base+l+m] for l, base in channels], dim=-1)), dim=1)
        if radial and front:
            pair = pair * radials[m].unsqueeze(1)
        raw = F.linear(pair, weights[m])
        real, imag = raw.chunk(2, dim=-1)
        yr, yi = real[:, 0] - imag[:, 1], real[:, 1] + imag[:, 0]
        if radial and not front:
            yr, yi = yr * radials[m], yi * radials[m]
        channels = [(l, start + c * (2*l+1)) for l, mul, start in outputs if l >= m for c in range(mul)]
        contribution = torch.zeros_like(actual)
        for i, (l, base) in enumerate(channels):
            contribution[:, base+l-m] = yr[:, i]
            contribution[:, base+l+m] = yi[:, i]
        expected = expected + contribution
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    differentiable = (x, *weights[1:]) + (tuple(radials[1:]) if radial else ())
    probe = torch.randn_like(actual)
    actual_grads = torch.autograd.grad((actual * probe).sum(), differentiable, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), differentiable)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=6e-5, rtol=6e-5)
