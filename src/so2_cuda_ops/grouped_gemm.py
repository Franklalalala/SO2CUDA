from __future__ import annotations

from typing import Optional

import torch


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
    flat_inputs = []
    for pair in pair_inputs:
        if pair.dim() != 3 or pair.size(1) != 2:
            raise RuntimeError("SO2 block-complex inputs must be [N, 2, Cin]")
        flat = pair.reshape(pair.shape[0], 2 * pair.shape[2])
        if permute_idx is not None:
            flat = flat.index_select(0, permute_idx)
        flat_inputs.append(flat.contiguous())

    flat_outputs = grouped_gemm_multi(flat_inputs, ptr_list, block_weights, fast_tf32=fast_tf32)
    outputs = []
    for flat_out, pair in zip(flat_outputs, pair_inputs):
        if unpermute_idx is not None:
            flat_out = flat_out.index_select(0, unpermute_idx)
        if flat_out.size(1) % 2 != 0:
            raise RuntimeError("SO2 block-complex output feature count must be even")
        cout = flat_out.size(1) // 2
        outputs.append(torch.stack((flat_out[:, :cout], flat_out[:, cout:]), dim=1).contiguous())
    return outputs
