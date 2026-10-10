# Changelog

## Unreleased

- Provide uniform and non-uniform SO(2) operator grids, with eager and compiled EquiformerV3, cuEquivariance, and our pure PyTorch reference implementation.
- Compare UniTB-dense, UniTB, and UniTB-SLEM on the same real training-batch streams for both onsite and hopping heads, using the same HybridMuon optimizer path across backends.
- Generate the English README performance tables and public numerical evidence from measurement JSON, including timing quartiles, peak memory, unsupported configurations, and out-of-memory results.
- Provide a complete tensor-only SO(2) example with irreps, edge vectors, trainable weights, and input/weight autograd.
- Document installation, DeePTB integration, supported inputs, reference fallback, and build settings in English.

## 0.3.1 (2026-10-11)

- Add edge-tiled rotation and gathering kernels. Each edge uses one thread block, staging input rows, Wigner blocks, and output rows in shared memory for contiguous access. Warps containing channels from different angular orders also combine memory accesses.
- Share rotation and gathering across top-k slots in `activation_forward`: one rotation writes slot-ordered copies, and one gathering sums gated slot outputs before rotating back. Backward also shares these operations. Expert group IDs come directly from routing results.
- Apply input-side radial weights during rotation and output-side radial weights during gathering. Radial layers save the input and recompute rotated blocks during backward; output-side radial layers also save the sum before radial scaling. This reduces saved activation memory.
- Add the synthetic UniTB-SLEM configuration [unitb_slem.json](examples/configs/unitb_slem.json).
- Preserve interface and numerical conventions.

## 0.3.0 (2026-10-10)

- Introduce block-layout sandwich kernels with one buffer per m. Positive and negative m components are paired; each complex linear map is represented as a real block matrix and evaluated with one GEMM. Rotation and gathering use kernels parallelized over edges and channels. Backward reuses these kernels without atomic additions.
- Add the keyword-only `include_m0` argument to `true_dense_pairs` and `dense_pairs`. With `True`, the m=0 term, including its bias, is computed in the same call. The default remains `False`.
- Use the same block-layout kernels for single-expert and top-k `activation_forward` and grouped `dense_pairs`, including radial weights and per-edge gates.
- Execute single-group FP32 GEMM with cuBLAS and calculate only requested gradients in grouped GEMM backward.
- Add operator comparisons against original EquiformerV3 (eager and `torch.compile`), our pure PyTorch implementation based on DeePTB upstream, and cuEquivariance. Check outputs, input gradients, parameter gradients, and rotation equivariance.
- Generate H200 performance tables from measurement JSON, recording strict FP32 timing and peak memory for operators and UniTB models.

## 0.2.0 (2026-10-07)

- Add tensor and layout-descriptor interfaces in `so2_cuda_ops.deeptb` for dense SO(2), activation-space expert routing, and grouped linear layers.
- Support UniTB-dense through `dense_pairs`, UniTB's PDQ-MoE through `activation_forward`, and non-MoE dense SO(2) through `true_dense_pairs`. DeePTB's `1006-stable` branch calls these through an optional backend.
- Maintain automatic differentiation for packing, grouped GEMM, scatter, and row permutations in SO2CUDA. Grouped GEMM supports JVP.
- Keep CUDA source in `src/so2_cuda_ops/csrc/`.
- Add synthetic periodic-structure examples for UniTB and UniTB-dense, comparing against our pure PyTorch implementation of upstream SO(2) tensor products and UMA-style expert linear layers. Record strict FP32 forward/backward timing, peak memory, and numerical differences.
