from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F


_INDEXED_MULTI_FLAT_BACKWARD_ENV = "DPTB_SO2_MOE_FUSED_P0_INDEXED_MULTI_FLAT_BACKWARD"


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("0", "false", "off", "no", "")


def _resolve_fast_tf32(fast_tf32: Optional[bool]) -> bool:
    if fast_tf32 is not None:
        return bool(fast_tf32)
    from so2_cuda_ops._cublas_grouped_gemm import _fast_tf32_enabled

    return bool(_fast_tf32_enabled())


def grouped_gemm(
    x: torch.Tensor,
    ptr: torch.Tensor,
    weight: torch.Tensor,
    *,
    fast_tf32: Optional[bool] = None,
) -> torch.Tensor:
    from so2_cuda_ops._cublas_grouped_gemm import grouped_gemm as _grouped_gemm

    return _grouped_gemm(x, ptr, weight, fast_tf32=fast_tf32)


def grouped_gemm_multi(
    xs: list[torch.Tensor],
    ptrs: list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    fast_tf32: Optional[bool] = None,
) -> list[torch.Tensor]:
    from so2_cuda_ops._cublas_grouped_gemm import grouped_gemm_multi as _grouped_gemm_multi

    return _grouped_gemm_multi(xs, ptrs, weights, fast_tf32=fast_tf32)


class _IndexedSandwichMultiFlatBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        ptr: torch.Tensor,
        permute_idx_or_empty: torch.Tensor,
        unpermute_idx_or_empty: torch.Tensor,
        raw_count: int,
        fast_tf32: bool,
        *pairs_and_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        raw_count = int(raw_count)
        pair_inputs = list(pairs_and_weights[:raw_count])
        weights = list(pairs_and_weights[raw_count:2 * raw_count])
        if len(pair_inputs) != raw_count or len(weights) != raw_count:
            raise RuntimeError("indexed_sandwich_multi flat backward received mismatched inputs")

        sorted_inputs: list[torch.Tensor] = []
        shapes: list[tuple[int, ...]] = []
        has_permute = permute_idx_or_empty.numel() > 0
        for pair in pair_inputs:
            shapes.append(tuple(pair.shape))
            flat = pair.reshape(-1, pair.shape[-1])
            if has_permute:
                flat = flat.index_select(0, permute_idx_or_empty)
            sorted_inputs.append(flat.contiguous())

        flat_outputs = grouped_gemm_multi(sorted_inputs, [ptr] * raw_count, weights, fast_tf32=fast_tf32)
        outputs: list[torch.Tensor] = []
        has_unpermute = unpermute_idx_or_empty.numel() > 0
        for flat_out, shape in zip(flat_outputs, shapes):
            if has_unpermute:
                flat_out = flat_out.index_select(0, unpermute_idx_or_empty)
            outputs.append(flat_out.reshape(*shape[:-1], flat_out.shape[-1]).contiguous())

        ctx.raw_count = raw_count
        ctx.fast_tf32 = bool(fast_tf32)
        ctx.shapes = shapes
        ctx.save_for_backward(
            ptr,
            permute_idx_or_empty,
            unpermute_idx_or_empty,
            *sorted_inputs,
            *weights,
        )
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        from so2_cuda_ops._cublas_grouped_gemm import _load_extension as _load_cublas_grouped

        raw_count = int(ctx.raw_count)
        tensors = ctx.saved_tensors
        ptr, permute_idx_or_empty, unpermute_idx_or_empty = tensors[:3]
        sorted_inputs = tensors[3:3 + raw_count]
        weights = tensors[3 + raw_count:3 + 2 * raw_count]
        ext = _load_cublas_grouped()

        has_permute = permute_idx_or_empty.numel() > 0
        has_unpermute = unpermute_idx_or_empty.numel() > 0
        sorted_grad_outputs: list[torch.Tensor] = []
        for grad_out in grad_outputs:
            grad_flat = grad_out.reshape(-1, grad_out.shape[-1]).contiguous()
            if has_permute:
                grad_sorted = grad_flat.index_select(0, permute_idx_or_empty).contiguous()
            else:
                grad_sorted = grad_flat
            sorted_grad_outputs.append(grad_sorted)

        ptrs = [ptr] * raw_count
        grad_x_sorted = ext.grouped_gemm_multi_forward_fp32(
            sorted_grad_outputs,
            ptrs,
            [weight.transpose(1, 2).contiguous() for weight in weights],
            ctx.fast_tf32,
        )
        grad_weights = ext.grouped_gemm_multi_backward_weight_fp32(
            sorted_grad_outputs,
            list(sorted_inputs),
            ptrs,
            ctx.fast_tf32,
        )

        grad_pairs: list[torch.Tensor] = []
        for grad_x, shape in zip(grad_x_sorted, ctx.shapes):
            if has_unpermute:
                grad_x_flat = grad_x.index_select(0, unpermute_idx_or_empty)
            else:
                grad_x_flat = grad_x
            grad_pairs.append(grad_x_flat.reshape(shape))

        return (None, None, None, None, None, *grad_pairs, *grad_weights)


def indexed_sandwich_multi_gemm(
    pair_inputs: list[torch.Tensor],
    ptrs: torch.Tensor | list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    permute_idx: torch.Tensor | None = None,
    unpermute_idx: torch.Tensor | None = None,
    fast_tf32: Optional[bool] = None,
) -> list[torch.Tensor]:
    """Shared middle GEMM for indexed_sandwich_multi-style SO2 paths."""
    if (
        _flag(_INDEXED_MULTI_FLAT_BACKWARD_ENV, False)
        and isinstance(ptrs, torch.Tensor)
    ):
        if len(pair_inputs) != len(weights):
            raise RuntimeError("indexed_sandwich_multi inputs and weights must have the same length")
        device = pair_inputs[0].device if pair_inputs else weights[0].device
        empty = torch.empty(0, dtype=torch.long, device=device)
        permute_arg = permute_idx if permute_idx is not None else empty
        unpermute_arg = unpermute_idx if unpermute_idx is not None else empty
        return list(
            _IndexedSandwichMultiFlatBackwardFunction.apply(
                ptrs,
                permute_arg,
                unpermute_arg,
                len(pair_inputs),
                _resolve_fast_tf32(fast_tf32),
                *pair_inputs,
                *weights,
            )
        )

    ptr_list = [ptrs] * len(pair_inputs) if isinstance(ptrs, torch.Tensor) else list(ptrs)
    flat_inputs = []
    for pair in pair_inputs:
        flat = pair.reshape(-1, pair.shape[-1])
        if permute_idx is not None:
            flat = flat.index_select(0, permute_idx)
        flat_inputs.append(flat.contiguous())

    flat_outputs = grouped_gemm_multi(flat_inputs, ptr_list, weights, fast_tf32=fast_tf32)
    outputs = []
    for flat_out, pair in zip(flat_outputs, pair_inputs):
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        outputs.append(flat_out.reshape(*pair.shape[:-1], flat_out.shape[-1]).contiguous())
    return outputs


def so2_block_complex_weights(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    """Build FairChem-style real/imag block weights from SO2 raw linear weights.

    Each input weight is shaped [G, 2*Cout, Cin] and represents W1/W2 rows used by
    SO2_m_Linear. The returned weight is [G, 2*Cout, 2*Cin] with rows
    [[W1, -W2], [W2, W1]], so a single GEMM maps [real, imag] input rows directly
    to finished [real, imag] pair outputs.
    """
    block_weights: list[torch.Tensor] = []
    for weight in weights:
        if weight.dim() != 3 or weight.size(1) % 2 != 0:
            raise RuntimeError("SO2 block-complex weights must be [G, 2*Cout, Cin]")
        cout = weight.size(1) // 2
        w_real = weight[:, :cout, :]
        w_imag = weight[:, cout:, :]
        top = torch.cat((w_real, -w_imag), dim=2)
        bottom = torch.cat((w_imag, w_real), dim=2)
        block_weights.append(torch.cat((top, bottom), dim=1).contiguous())
    return block_weights


def indexed_sandwich_multi_block_gemm(
    pair_inputs: list[torch.Tensor],
    ptrs: torch.Tensor | list[torch.Tensor],
    weights: list[torch.Tensor],
    *,
    permute_idx: torch.Tensor | None = None,
    unpermute_idx: torch.Tensor | None = None,
    fast_tf32: Optional[bool] = None,
) -> list[torch.Tensor]:
    """SO2 m>0 middle GEMM using a larger real/imag block matrix."""
    ptr_list = [ptrs] * len(pair_inputs) if isinstance(ptrs, torch.Tensor) else list(ptrs)
    block_weights = so2_block_complex_weights(weights)
    if len(ptr_list) != len(pair_inputs) or len(block_weights) != len(pair_inputs):
        raise RuntimeError("SO2 block-complex inputs, ptrs, and weights must have the same length")
    flat_inputs = []
    full_single_group = permute_idx is None and unpermute_idx is None
    for pair, ptr, weight in zip(pair_inputs, ptr_list, block_weights):
        if pair.dim() != 3 or pair.size(1) != 2:
            raise RuntimeError("SO2 block-complex inputs must be [N, 2, Cin]")
        flat = pair.reshape(pair.shape[0], 2 * pair.shape[2])
        if (
            full_single_group
            and weight.size(0) == 1
            and not ptr.is_cuda
            and ptr.numel() == 2
            and int(ptr[0].item()) == 0
            and int(ptr[1].item()) == int(pair.shape[0])
        ):
            pass
        else:
            full_single_group = False
        if permute_idx is not None:
            flat = flat.index_select(0, permute_idx)
        flat_inputs.append(flat.contiguous())

    if full_single_group:
        flat_outputs = [
            F.linear(flat, weight.squeeze(0))
            for flat, weight in zip(flat_inputs, block_weights)
        ]
    else:
        flat_outputs = grouped_gemm_multi(flat_inputs, ptr_list, block_weights, fast_tf32=fast_tf32)
    outputs = []
    for flat_out, pair in zip(flat_outputs, pair_inputs):
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        if flat_out.size(1) % 2 != 0:
            raise RuntimeError("SO2 block-complex output feature count must be even")
        cout = flat_out.size(1) // 2
        outputs.append(flat_out.reshape(flat_out.shape[0], 2, cout).contiguous())
    return outputs


class _BlockComplexDirectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pair: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        from so2_cuda_ops.tensor_product import _load_extension

        pair_c = pair.contiguous()
        weight_c = weight.contiguous()
        out = _load_extension().block_complex_forward_fp32(pair_c, weight_c)
        ctx.save_for_backward(pair_c, weight_c)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        from so2_cuda_ops.tensor_product import _load_extension

        pair, weight = ctx.saved_tensors
        grad_pair, grad_weight = _load_extension().block_complex_backward_fp32(
            grad_out.contiguous(),
            pair,
            weight,
        )
        return grad_pair, grad_weight


def block_complex_direct(pair: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Direct SO2 block-complex pair GEMM without materializing W_block."""
    return _BlockComplexDirectFunction.apply(pair, weight)


def indexed_sandwich_multi_block_direct_gemm(
    pair_inputs: list[torch.Tensor],
    weights: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Apply compact W1/W2 block-complex GEMM for each m without W_block."""
    if len(pair_inputs) != len(weights):
        raise RuntimeError("SO2 direct block-complex inputs and weights must have the same length")
    return [
        block_complex_direct(pair, weight)
        for pair, weight in zip(pair_inputs, weights)
    ]
