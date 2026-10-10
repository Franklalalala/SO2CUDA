"""Autograd buffer lifetime and edge-kernel shared-memory boundaries."""
import pytest
import torch

from so2_cuda_ops import _sandwich
from so2_cuda_ops.deeptb import (
    ActivationRouting, LinearWeights, activation_forward, prepare_layout, prepare_wigner,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("front", [True, False])
def test_radial_recompute_checks_wigner_version(front):
    x = torch.tensor([[0., 1., 0.]], device="cuda")
    layout = prepare_layout(((1, 1, 0),), ((0, 1, 0),), m_max=0, l_max=1,
                            out_dim=1, device="cuda", rotate_out=False, front=front)
    blocks = (torch.ones(1, 1, 1, device="cuda"), torch.eye(3, device="cuda")[None])
    wigner = prepare_wigner(x, blocks, l_max=1)
    weight = torch.ones(1, 1, device="cuda", requires_grad=True)
    y = _sandwich.full_sandwich(x, layout, wigner, weight, None, (None,),
                                (torch.ones(1, 1, device="cuda"),))
    assert y.grad_fn.state[3].fused_radial
    torch.testing.assert_close(y, torch.ones_like(y))
    wigner.values.mul_(-1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        y.sum().backward()


@pytest.mark.parametrize("radial", [None, "front", "back"])
@pytest.mark.parametrize("gradient", ["bias", "gate"])
@pytest.mark.parametrize("external_ids", [True, False])
def test_sorted_group_ids_version_check(radial, gradient, external_ids):
    x = torch.ones(2, 1, device="cuda")
    layout = prepare_layout(((0, 1, 0),), ((0, 1, 0),), m_max=0, l_max=0,
                            out_dim=1, device="cuda", rotate_in=False, rotate_out=False,
                            front=radial != "back")
    wigner = prepare_wigner(x, None, l_max=0, rotate=False)
    bias = torch.tensor([[2.], [3.]], device="cuda", requires_grad=gradient == "bias")
    gates = torch.ones(2, 1, device="cuda", requires_grad=gradient == "gate")
    ids = torch.tensor([0, 1], device="cuda")
    order = torch.arange(2, device="cuda")
    slot = (order, order, torch.tensor([0, 1, 2])) + ((ids,) if external_ids else ())
    routing = ActivationRouting(ids[:, None], gates, (slot,))
    linears = (LinearWeights(torch.zeros(2, 1, 1, device="cuda"), bias),)
    radials = (torch.ones_like(x),) if radial else None
    y = activation_forward(x, layout, wigner, linears, radials, routing)
    torch.testing.assert_close(y, bias.detach())
    ids.copy_(torch.tensor([1, 0], device="cuda"))
    loss = (y[:, 0] * torch.tensor([1., 2.], device="cuda")).sum()
    if external_ids:
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            loss.backward()
    else:
        loss.backward()
        actual = bias.grad if gradient == "bias" else gates.grad
        expected = [1., 2.] if gradient == "bias" else [2., 6.]
        torch.testing.assert_close(actual[:, 0], torch.tensor(expected, device="cuda"))


@pytest.mark.parametrize("boundary", ["default", "optin"])
@pytest.mark.parametrize("radial", [False, True])
def test_gather_static_shared_memory_boundary(boundary, radial):
    limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    width = 6144 if boundary == "default" else limit // 8
    x = torch.tensor([[1.], [2.]], device="cuda", requires_grad=True)
    layout = prepare_layout(((0, 1, 0),), ((0, width, 0),), m_max=0, l_max=0,
                            out_dim=width, device="cuda", rotate_in=False, rotate_out=False,
                            front=False)
    wigner = prepare_wigner(x, None, l_max=0, rotate=False)
    w = torch.ones(width, 1, device="cuda", requires_grad=True)
    r = torch.full((2, width), 0.5, device="cuda", requires_grad=True) if radial else None
    y = _sandwich.full_sandwich(x, layout, wigner, w, None, (None,), (r,) if radial else None)
    if radial and boundary == "optin":
        assert not y.grad_fn.state[3].fused_radial
    scale = 0.5 if radial else 1.
    torch.testing.assert_close(y, x.detach().expand(-1, width) * scale)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, width * scale))
    torch.testing.assert_close(w.grad, torch.full_like(w, 3 * scale))
    if radial:
        torch.testing.assert_close(r.grad, x.detach().expand(-1, width))
