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
