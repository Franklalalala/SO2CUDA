"""Rotated m>0 pair contributions of the public pair APIs against a float64 reference."""
import pytest
import torch

from so2_cuda_ops.deeptb import (
    DenseRouting, LinearWeights, dense_pairs, prepare_layout, prepare_wigner, true_dense_pairs,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

V_IN = ((0, 5, 0), (1, 3, 5), (2, 4, 14), (3, 2, 34), (1, 2, 48))
V_OUT = ((0, 4, 0), (1, 2, 4), (2, 3, 10), (3, 3, 25), (2, 1, 46))


def _dim(entries):
    return max(start + mul * (2 * l + 1) for l, mul, start in entries)


def _blocks(n, l_max, generator):
    """Orthogonal per-l blocks; the kernels only need per-l matrices."""
    blocks = []
    for l in range(l_max + 1):
        q, _ = torch.linalg.qr(torch.randn(n, 2 * l + 1, 2 * l + 1, generator=generator, dtype=torch.float64))
        blocks.append(q)
    return blocks


def _channels(entries, m):
    return [(l, start + c * (2 * l + 1)) for l, mul, start in entries if l >= m for c in range(mul)]


def _reference(x, blocks, entries_in, entries_out, weights, radials, front, out_dim):
    """Sum over m>0 of D^T W_m D x per edge, in float64."""
    n = x.shape[0]
    out = torch.zeros(n, out_dim, dtype=torch.float64)
    for m, weight in enumerate(weights):
        if m == 0:
            continue
        cin = _channels(entries_in, m)
        cout = _channels(entries_out, m)
        if not cin or not cout:
            continue
        rot = [torch.einsum("nd,ndj->nj", x[:, b:b + 2 * l + 1], blocks[l]) for l, b in cin]
        pair = torch.stack((torch.stack([r[:, l - m] for r, (l, _) in zip(rot, cin)], -1),
                            torch.stack([r[:, l + m] for r, (l, _) in zip(rot, cin)], -1)), dim=1)
        if radials is not None and front:
            pair = pair * radials[m].unsqueeze(1)
        raw = pair @ weight.t()
        a, b = raw.chunk(2, dim=-1)
        y = torch.stack((a[:, 0] - b[:, 1], a[:, 1] + b[:, 0]), dim=1)
        if radials is not None and not front:
            y = y * radials[m].unsqueeze(1)
        for c, (l, base) in enumerate(cout):
            local = torch.zeros(n, 2 * l + 1, dtype=torch.float64)
            local[:, l - m] = y[:, 0, c]
            local[:, l + m] = y[:, 1, c]
            out[:, base:base + 2 * l + 1] += torch.einsum("nj,nij->ni", local, blocks[l])
    return out


HIGH_IN = ((0, 2, 0), (9, 2, 2), (4, 3, 40))
HIGH_OUT = ((1, 2, 0), (9, 1, 6), (5, 2, 25))


def _case(front, radial, seed=7, n=33, entries=(V_IN, V_OUT), m_max=None):
    generator = torch.Generator().manual_seed(seed)
    entries_in, entries_out = entries if front else entries[::-1]
    dim_in, dim_out = _dim(entries_in), _dim(entries_out)
    l_max = max(l for l, _, _ in entries_in + entries_out)
    m_max = l_max if m_max is None else m_max
    x = torch.randn(n, dim_in, generator=generator, dtype=torch.float64)
    blocks = _blocks(n, l_max, generator)
    weights, radials = [], []
    for m in range(m_max + 1):
        cin = sum(mul for l, mul, _ in entries_in if l >= m)
        cout = sum(mul for l, mul, _ in entries_out if l >= m)
        weights.append(torch.randn(cout * (2 if m else 1), cin, generator=generator, dtype=torch.float64))
        radials.append(torch.randn(n, cin if front else cout, generator=generator, dtype=torch.float64))
    return entries_in, entries_out, dim_in, dim_out, l_max, m_max, x, blocks, weights, (radials if radial else None)


def _check(fn, front, radial, **case):
    entries_in, entries_out, dim_in, dim_out, l_max, m_max, x64, blocks64, w64, r64 = _case(front, radial, **case)
    probe = torch.randn(x64.shape[0], dim_out, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    leaves64 = [x64.clone().requires_grad_(True)] + [w.clone().requires_grad_(True) for w in w64[1:]]
    rad64 = None if r64 is None else [r.clone().requires_grad_(True) for r in r64]
    ref = _reference(leaves64[0], blocks64, entries_in, entries_out, [w64[0]] + leaves64[1:], rad64, front, dim_out)
    ref_grads = torch.autograd.grad((ref * probe).sum(), leaves64 + (rad64[1:] if rad64 else []))

    dev = "cuda"
    x = leaves64[0].detach().float().to(dev).requires_grad_(True)
    ws = [w.detach().float().to(dev).requires_grad_(True) for w in w64]
    rs = None if r64 is None else [r.detach().float().to(dev).requires_grad_(True) for r in r64]
    layout = prepare_layout(entries_in, entries_out, m_max=m_max, l_max=l_max, out_dim=dim_out,
                            device=dev, front=front)
    wigner = prepare_wigner(x, tuple(b.float().to(dev) for b in blocks64), l_max=l_max)
    parts = fn(x, layout, wigner, ws, rs)
    assert parts is not None
    actual = sum(parts)
    grads = torch.autograd.grad((actual * probe.float().to(dev)).sum(),
                                [x] + ws[1:] + (rs[1:] if rs else []))
    scale = ref.detach().abs().max()
    torch.testing.assert_close(actual.double().cpu(), ref.detach(), atol=2e-5 * float(scale), rtol=0)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a.double().cpu(), b, atol=3e-5 * float(b.abs().max()) + 1e-6, rtol=0)


def _true_dense(x, layout, wigner, ws, rs):
    return true_dense_pairs(x, layout, wigner, tuple(LinearWeights(w, routed=False) for w in ws),
                            None if rs is None else tuple(rs))


def _dense_single_group(x, layout, wigner, ws, rs):
    n = x.shape[0]
    routing = DenseRouting(torch.zeros(n, dtype=torch.long, device=x.device), torch.tensor([0, 2 * n]))
    return dense_pairs(x, layout, wigner, tuple(w.unsqueeze(0) for w in ws),
                       None if rs is None else tuple(rs), routing)


@pytest.mark.parametrize("front", [True, False])
@pytest.mark.parametrize("radial", [True, False])
def test_true_dense_pairs_rotated(front, radial):
    _check(_true_dense, front, radial)


@pytest.mark.parametrize("front", [True, False])
@pytest.mark.parametrize("radial", [True, False])
def test_dense_pairs_single_group_rotated(front, radial):
    _check(_dense_single_group, front, radial)


@pytest.mark.parametrize("front", [True, False])
def test_true_dense_pairs_degrees_above_unrolled_range(front):
    _check(_true_dense, front, True, entries=(HIGH_IN, HIGH_OUT))


@pytest.mark.parametrize("m_max", [1, 2])
def test_pairs_truncated_m(m_max):
    _check(_true_dense, True, False, m_max=m_max)
    _check(_dense_single_group, True, True, m_max=m_max)
