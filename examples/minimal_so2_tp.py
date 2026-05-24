import torch

from so2_cuda_ops import indexed_sandwich_multi


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example")

    pair = torch.randn(11, 2, 5, device="cuda", dtype=torch.float32)
    ptr = torch.tensor([0, 7, 14, 22], dtype=torch.long, device="cuda")
    weight = torch.randn(3, 4, 5, device="cuda", dtype=torch.float32)
    out = indexed_sandwich_multi([pair], ptr, [weight])[0]
    print(tuple(out.shape))


if __name__ == "__main__":
    main()
