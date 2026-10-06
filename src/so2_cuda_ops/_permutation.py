"""Differentiable row permutations shared by SO2 and grouped linears."""
import torch

class _RowPermutation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order, inverse):
        ctx.save_for_backward(order, inverse)
        return x.index_select(0, order)

    @staticmethod
    def backward(ctx, grad):
        order, inverse = ctx.saved_tensors
        return permute_rows(grad, inverse, order), None, None


def permute_rows(x: torch.Tensor, order: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
    """x[order] for a permutation ``order`` of the rows of x whose inverse is ``inverse``.

    The backward gathers the gradient with ``inverse``, where index_select's backward
    zero-fills a buffer and index_adds into it.  Values and gradients are those of
    index_select; only a bijection qualifies (a gather with repeated rows must sum).
    Under a torch.func transform this is index_select itself."""
    if getattr(torch._C, "_are_functorch_transforms_active", lambda: False)():
        return x.index_select(0, order)
    return _RowPermutation.apply(x, order, inverse)

