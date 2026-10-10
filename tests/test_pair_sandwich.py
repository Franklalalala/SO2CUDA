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


def _reference_layer(x, blocks, entries_in, entries_out, weights, bias0, radials, front, out_dim, gate):
    """gate * (m = 0 term + m > 0 terms) of one SO(2) layer, in float64."""
    n = x.shape[0]
    out = _reference(x, blocks, entries_in, entries_out, weights, radials, front, out_dim)
    cin = _channels(entries_in, 0)
    cout = _channels(entries_out, 0)
    x0 = torch.stack([torch.einsum("nd,nd->n", x[:, b:b + 2 * l + 1], blocks[l][:, :, l]) for l, b in cin], -1)
    if radials is not None and front:
        x0 = x0 * radials[0]
    y0 = x0 @ weights[0].t() + bias0
    if radials is not None and not front:
        y0 = y0 * radials[0]
    for c, (l, base) in enumerate(cout):
        out[:, base:base + 2 * l + 1] += y0[:, c:c + 1] * blocks[l][:, :, l]
    return out * gate.unsqueeze(1)


@pytest.mark.parametrize("radial", [None, "front", "back"])
def test_activation_single_expert_rotated(radial):
    from so2_cuda_ops.deeptb import ActivationRouting, activation_forward
    front = radial != "back"
    entries_in, entries_out, dim_in, dim_out, l_max, m_max, x64, blocks64, w64, r64 = _case(
        front, radial is not None)
    n = x64.shape[0]
    g = torch.Generator().manual_seed(11)
    b64 = torch.randn(w64[0].shape[0], generator=g, dtype=torch.float64)
    gate64 = torch.rand(n, generator=g, dtype=torch.float64) + 0.5
    probe = torch.randn(n, dim_out, generator=g, dtype=torch.float64)
    leaves = [x64, b64, gate64] + list(w64) + (list(r64) if r64 is not None else [])
    leaves = [t.clone().requires_grad_(True) for t in leaves]
    xr, br, gr, wr, rr = leaves[0], leaves[1], leaves[2], leaves[3:3 + len(w64)], leaves[3 + len(w64):]
    ref = _reference_layer(xr, blocks64, entries_in, entries_out, wr, br, rr or None, front, dim_out, gr)
    ref_grads = torch.autograd.grad((ref * probe).sum(), leaves)

    dev = "cuda"
    cuda = [t.detach().float().to(dev).requires_grad_(True) for t in leaves]
    x, b0, gate, ws, rs = cuda[0], cuda[1], cuda[2], cuda[3:3 + len(w64)], cuda[3 + len(w64):]
    layout = prepare_layout(entries_in, entries_out, m_max=m_max, l_max=l_max, out_dim=dim_out,
                            device=dev, front=front)
    wigner = prepare_wigner(x, tuple(b.float().to(dev) for b in blocks64), l_max=l_max)
    linears = (LinearWeights(ws[0].unsqueeze(0), b0.unsqueeze(0)),) + tuple(
        LinearWeights(w.unsqueeze(0)) for w in ws[1:])
    radials = None
    if rs:
        radials = (rs[0], torch.cat(rs[1:], dim=-1)) if front else tuple(rs)
    idx = torch.zeros(n, 1, dtype=torch.long, device=dev)
    order = torch.arange(n, device=dev)
    slot = (order, order, torch.tensor([0, n]), idx[:, 0])
    routing = ActivationRouting(idx, gate.unsqueeze(1), (slot,))
    actual = activation_forward(x, layout, wigner, linears, radials, routing)
    assert actual is not None
    grads = torch.autograd.grad((actual * probe.float().to(dev)).sum(), cuda)
    scale = ref.detach().abs().max()
    torch.testing.assert_close(actual.double().cpu(), ref.detach(), atol=2e-5 * float(scale), rtol=0)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a.double().cpu(), b, atol=3e-5 * float(b.abs().max()) + 1e-6, rtol=0)


def _grouped_reference(x, blocks, entries_in, entries_out, weights, radials, front, out_dim, graph):
    """Sum over groups of the m>0 terms of each group's edges with that group's weights."""
    out = torch.zeros(x.shape[0], out_dim, dtype=torch.float64)
    for g in range(weights[1].shape[0]):
        sel = (graph == g).nonzero().flatten()
        if sel.numel() == 0:
            continue
        sub_blocks = [b[sel] for b in blocks]
        sub_radials = None if radials is None else [r[sel] for r in radials]
        part = _reference(x[sel], sub_blocks, entries_in, entries_out,
                          [weights[0]] + [w[g] for w in weights[1:]], sub_radials, front, out_dim)
        out = out.index_add(0, sel, part)
    return out


@pytest.mark.parametrize("front", [True, False])
@pytest.mark.parametrize("radial", [True, False])
def test_dense_pairs_grouped_rotated(front, radial):
    entries_in, entries_out, dim_in, dim_out, l_max, m_max, x64, blocks64, w64, r64 = _case(front, radial)
    n = x64.shape[0]
    gen = torch.Generator().manual_seed(5)
    groups = 3
    graph = torch.randint(0, groups, (n,), generator=gen)
    graph[0] = 1  # not already sorted
    wg64 = [w64[0]] + [torch.randn(groups, *w.shape, generator=gen, dtype=torch.float64) for w in w64[1:]]
    probe = torch.randn(n, dim_out, generator=gen, dtype=torch.float64)
    leaves = [x64] + wg64[1:] + (list(r64[1:]) if r64 is not None else [])
    leaves = [t.clone().requires_grad_(True) for t in leaves]
    xr, wr, rr = leaves[0], leaves[1:len(wg64)], leaves[len(wg64):]
    ref = _grouped_reference(xr, blocks64, entries_in, entries_out, [wg64[0]] + wr,
                             [r64[0]] + rr if rr else None, front, dim_out, graph)
    ref_grads = torch.autograd.grad((ref * probe).sum(), leaves)

    dev = "cuda"
    cuda = [t.detach().float().to(dev).requires_grad_(True) for t in leaves]
    x, ws, rs = cuda[0], cuda[1:len(wg64)], cuda[len(wg64):]
    layout = prepare_layout(entries_in, entries_out, m_max=m_max, l_max=l_max, out_dim=dim_out,
                            device=dev, front=front)
    wigner = prepare_wigner(x, tuple(b.float().to(dev) for b in blocks64), l_max=l_max)
    graph_dev = graph.to(dev)
    flat = graph_dev.repeat_interleave(2)
    permute = torch.argsort(flat, stable=True)
    unpermute = torch.argsort(permute)
    counts = torch.bincount(flat, minlength=groups).cpu()
    ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    routing = DenseRouting(graph_dev, ptr, permute, unpermute)
    weights = (torch.zeros(1, 0, 0, device=dev),) + tuple(ws)
    radials = None if not rs else (torch.zeros(n, 0, device=dev),) + tuple(rs)
    parts = dense_pairs(x, layout, wigner, weights, radials, routing)
    assert parts is not None
    actual = sum(parts)
    grads = torch.autograd.grad((actual * probe.float().to(dev)).sum(), cuda)
    scale = ref.detach().abs().max()
    torch.testing.assert_close(actual.double().cpu(), ref.detach(), atol=2e-5 * float(scale), rtol=0)
    for a, b in zip(grads, ref_grads):
        if b.abs().max() == 0:
            continue
        torch.testing.assert_close(a.double().cpu(), b, atol=3e-5 * float(b.abs().max()) + 1e-6, rtol=0)


@pytest.mark.parametrize("radial", [None, "front", "back"])
def test_activation_routed_experts_rotated(radial):
    from so2_cuda_ops.deeptb import ActivationRouting, activation_forward
    front = radial != "back"
    entries_in, entries_out, dim_in, dim_out, l_max, m_max, x64, blocks64, w64, r64 = _case(
        front, radial is not None)
    n = x64.shape[0]
    experts, k = 3, 2
    g = torch.Generator().manual_seed(13)
    idx = torch.stack([torch.randperm(experts, generator=g)[:k] for _ in range(n)])
    we64 = [torch.randn(experts, *w.shape, generator=g, dtype=torch.float64) for w in w64]
    be64 = torch.randn(experts, w64[0].shape[0], generator=g, dtype=torch.float64)
    gate64 = torch.rand(n, k, generator=g, dtype=torch.float64) + 0.25
    probe = torch.randn(n, dim_out, generator=g, dtype=torch.float64)
    leaves = [x64, be64, gate64] + we64 + (list(r64) if r64 is not None else [])
    leaves = [t.clone().requires_grad_(True) for t in leaves]
    xr, br, gr, wr, rr = leaves[0], leaves[1], leaves[2], leaves[3:3 + len(we64)], leaves[3 + len(we64):]
    ref = torch.zeros(n, dim_out, dtype=torch.float64)
    for j in range(k):
        for e in range(experts):
            sel = (idx[:, j] == e).nonzero().flatten()
            if sel.numel() == 0:
                continue
            part = _reference_layer(xr[sel], [b[sel] for b in blocks64], entries_in, entries_out,
                                    [w[e] for w in wr], br[e], [r[sel] for r in rr] if rr else None, front,
                                    dim_out, gr[sel, j])
            ref = ref.index_add(0, sel, part)
    ref_grads = torch.autograd.grad((ref * probe).sum(), leaves)

    dev = "cuda"
    cuda = [t.detach().float().to(dev).requires_grad_(True) for t in leaves]
    x, b0, gate, ws, rs = cuda[0], cuda[1], cuda[2], cuda[3:3 + len(we64)], cuda[3 + len(we64):]
    layout = prepare_layout(entries_in, entries_out, m_max=m_max, l_max=l_max, out_dim=dim_out,
                            device=dev, front=front)
    wigner = prepare_wigner(x, tuple(b.float().to(dev) for b in blocks64), l_max=l_max)
    linears = (LinearWeights(ws[0], b0),) + tuple(LinearWeights(w) for w in ws[1:])
    radials = None
    if rs:
        radials = (rs[0], torch.cat(rs[1:], dim=-1)) if front else tuple(rs)
    idx_dev = idx.to(dev)
    slots = []
    for j in range(k):
        order = torch.argsort(idx_dev[:, j], stable=True)
        inverse = torch.argsort(order)
        counts = torch.bincount(idx_dev[:, j], minlength=experts).cpu()
        ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
        slots.append((order, inverse, ptr, idx_dev[:, j][order]))
    routing = ActivationRouting(idx_dev, gate, tuple(slots), coefficients_sum_to_one=False)
    actual = activation_forward(x, layout, wigner, linears, radials, routing)
    assert actual is not None
    grads = torch.autograd.grad((actual * probe.float().to(dev)).sum(), cuda)
    scale = ref.detach().abs().max()
    torch.testing.assert_close(actual.double().cpu(), ref.detach(), atol=2e-5 * float(scale), rtol=0)
    for a, b in zip(grads, ref_grads):
        torch.testing.assert_close(a.double().cpu(), b, atol=3e-5 * float(b.abs().max()) + 1e-6, rtol=0)
