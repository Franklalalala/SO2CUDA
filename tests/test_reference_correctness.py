import pytest

torch = pytest.importorskip("torch")


def _segment_reference(x, ptr, weight):
    out = torch.empty((x.shape[0], weight.shape[1]), dtype=x.dtype, device=x.device)
    for group in range(weight.shape[0]):
        start = int(ptr[group].item())
        end = int(ptr[group + 1].item())
        if end > start:
            out[start:end] = x[start:end].matmul(weight[group].transpose(0, 1))
    return out


def test_indexed_sandwich_multi_matches_torch_reference_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA grouped GEMM correctness requires CUDA")

    from so2_cuda_ops import indexed_sandwich_multi

    torch.manual_seed(20260524)
    pair = torch.randn(11, 2, 5, device="cuda", dtype=torch.float32)
    ptr = torch.tensor([0, 7, 14, 22], dtype=torch.long, device="cuda")
    weight = torch.randn(3, 4, 5, device="cuda", dtype=torch.float32)

    actual = indexed_sandwich_multi([pair], ptr, [weight])[0]
    expected = _segment_reference(pair.reshape(-1, 5), ptr.cpu(), weight).reshape(11, 2, 4)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_indexed_sandwich_multi_block_gemm_matches_raw_finish_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("SO2 block GEMM correctness requires CUDA")

    from so2_cuda_ops import indexed_sandwich_multi_block_gemm, indexed_sandwich_multi_gemm

    torch.manual_seed(20260525)
    pair = torch.randn(13, 2, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(1, 8, 6, device="cuda", dtype=torch.float32, requires_grad=True)
    ptr_pair_rows = torch.tensor([0, 13], dtype=torch.long)
    ptr_raw_rows = torch.tensor([0, 26], dtype=torch.long)

    raw = indexed_sandwich_multi_gemm([pair], ptr_raw_rows, [weight])[0]
    raw_r = raw[:, :, :4]
    raw_i = raw[:, :, 4:]
    expected = torch.cat(
        (
            raw_r.narrow(1, 0, 1) - raw_i.narrow(1, 1, 1),
            raw_r.narrow(1, 1, 1) + raw_i.narrow(1, 0, 1),
        ),
        dim=1,
    )
    actual = indexed_sandwich_multi_block_gemm([pair], ptr_pair_rows, [weight])[0]

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    grad = torch.randn_like(expected)
    expected.backward(grad, retain_graph=True)
    pair_grad_expected = pair.grad.detach().clone()
    weight_grad_expected = weight.grad.detach().clone()
    pair.grad = None
    weight.grad = None
    actual.backward(grad)

    torch.testing.assert_close(pair.grad, pair_grad_expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(weight.grad, weight_grad_expected, atol=3e-5, rtol=3e-5)


def test_indexed_sandwich_multi_block_direct_matches_raw_finish_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("SO2 direct block-complex correctness requires CUDA")

    from so2_cuda_ops import indexed_sandwich_multi_block_direct_gemm, indexed_sandwich_multi_gemm

    torch.manual_seed(20260525)
    pair = torch.randn(17, 2, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(1, 10, 7, device="cuda", dtype=torch.float32, requires_grad=True)
    ptr_raw_rows = torch.tensor([0, 34], dtype=torch.long)

    raw = indexed_sandwich_multi_gemm([pair], ptr_raw_rows, [weight])[0]
    raw_r = raw[:, :, :5]
    raw_i = raw[:, :, 5:]
    expected = torch.cat(
        (
            raw_r.narrow(1, 0, 1) - raw_i.narrow(1, 1, 1),
            raw_r.narrow(1, 1, 1) + raw_i.narrow(1, 0, 1),
        ),
        dim=1,
    )
    actual = indexed_sandwich_multi_block_direct_gemm([pair], [weight])[0]

    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)

    grad = torch.randn_like(expected)
    expected.backward(grad, retain_graph=True)
    pair_grad_expected = pair.grad.detach().clone()
    weight_grad_expected = weight.grad.detach().clone()
    pair.grad = None
    weight.grad = None
    actual.backward(grad)

    torch.testing.assert_close(pair.grad, pair_grad_expected, atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(weight.grad, weight_grad_expected, atol=5e-5, rtol=5e-5)


def test_raw_pairs_multi_output_grad_matches_per_m_reference_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("SO2 raw multi output grad correctness requires CUDA")

    from so2_cuda_ops.tensor_product import (
        _raw_pair_output_grad_cuda,
        _raw_pairs_multi_output_grad_cuda,
    )

    torch.manual_seed(20260525)
    grad_out = torch.randn(17, 9, device="cuda", dtype=torch.float32)
    wigner = torch.empty(0, device="cuda", dtype=torch.float32)
    offsets = torch.zeros(3, device="cuda", dtype=torch.long)
    compact_offsets = torch.empty(0, device="cuda", dtype=torch.long)
    out_bases = [
        torch.tensor([0, 3], device="cuda", dtype=torch.long),
        torch.tensor([3], device="cuda", dtype=torch.long),
    ]
    out_ls = [
        torch.tensor([1, 2], device="cuda", dtype=torch.long),
        torch.tensor([2], device="cuda", dtype=torch.long),
    ]
    cout_prefix = torch.tensor([0, 2, 3], device="cuda", dtype=torch.long)
    m_values = torch.tensor([1, 2], device="cuda", dtype=torch.long)

    expected = [
        _raw_pair_output_grad_cuda(
            grad_out,
            wigner,
            out_base,
            out_l,
            offsets,
            compact_offsets,
            int(m),
            False,
            0,
            0,
        )
        for m, out_base, out_l in zip((1, 2), out_bases, out_ls)
    ]
    actual = _raw_pairs_multi_output_grad_cuda(
        grad_out,
        wigner,
        out_bases,
        out_ls,
        offsets,
        compact_offsets,
        cout_prefix,
        m_values,
        False,
        0,
        0,
    )

    assert len(actual) == len(expected)
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)


def test_pack_pairs_multi_desc_matches_pointer_multi_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("SO2 descriptor pack correctness requires CUDA")

    from so2_cuda_ops.tensor_product import _pack_pairs_multi_cuda, _pack_pairs_multi_desc_cuda

    torch.manual_seed(20260525)
    x = torch.randn(19, 12, device="cuda", dtype=torch.float32)
    wigner = torch.empty(0, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 4], device="cuda", dtype=torch.long)
    compact_offsets = torch.empty(0, device="cuda", dtype=torch.long)
    in_bases = [
        torch.tensor([1, 4, 7], device="cuda", dtype=torch.long),
        torch.tensor([4], device="cuda", dtype=torch.long),
    ]
    in_ls = [
        torch.tensor([1, 1, 2], device="cuda", dtype=torch.long),
        torch.tensor([2], device="cuda", dtype=torch.long),
    ]
    cin_prefix = torch.tensor([0, 3, 4], device="cuda", dtype=torch.long)
    m_values = torch.tensor([1, 2], device="cuda", dtype=torch.long)
    desc = torch.tensor(
        [
            [1, 1, 1],
            [4, 1, 1],
            [7, 2, 1],
            [4, 2, 2],
        ],
        device="cuda",
        dtype=torch.long,
    )

    expected = _pack_pairs_multi_cuda(
        x,
        wigner,
        in_bases,
        in_ls,
        offsets,
        compact_offsets,
        cin_prefix,
        m_values,
        False,
        0,
        0,
    )
    actual = _pack_pairs_multi_desc_cuda(
        x,
        wigner,
        desc,
        offsets,
        compact_offsets,
        False,
        0,
        0,
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_scatter_m0_forward_matches_reference_if_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("SO2 m0 scatter correctness requires CUDA")

    from so2_cuda_ops.tensor_product import _scatter_m0_forward_cuda

    torch.manual_seed(20260525)
    y_m0 = torch.randn(13, 3, device="cuda", dtype=torch.float32)
    wigner = torch.empty(0, device="cuda", dtype=torch.float32)
    offsets = torch.tensor([0, 1, 4], device="cuda", dtype=torch.long)
    compact_offsets = torch.empty(0, device="cuda", dtype=torch.long)
    out_base = torch.tensor([0, 3, 6], device="cuda", dtype=torch.long)
    out_l = torch.tensor([1, 1, 2], device="cuda", dtype=torch.long)

    actual = _scatter_m0_forward_cuda(
        y_m0,
        wigner,
        out_base,
        out_l,
        offsets,
        compact_offsets,
        11,
        False,
        0,
        0,
    )
    expected = y_m0.new_zeros((13, 11))
    for c, (base, l) in enumerate(zip(out_base.cpu().tolist(), out_l.cpu().tolist())):
        expected[:, base + l] = y_m0[:, c]

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
