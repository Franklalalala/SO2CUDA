# Extraction Inventory

Moved into this package:

- `cublas_grouped_gemm.cpp` and Python grouped GEMM facade.
- SO2 pack/scatter and indexed sandwich extension sources.
- SO2 materialized scheduler native extension sources.
- Shared CUDA JIT loader and segment layout helpers.
- Optional CUTLASS grouped GEMM and SO2 smoke sources.

Kept in DeePTB:

- `SO2_Linear` model classes and mode normalization.
- DeePTB training/config schema and old `DPTB_SO2_*` environment compatibility.
- Shape gates, fallback policy, and model-specific `block_dense` orchestration.
- End-to-end DeePTB tests and benchmarks.

Dropped from DeePTB:

- Embedded CUDA/C++ implementation sources under `dptb/nn/csrc`.

The DeePTB files that previously owned backend implementation are now import shims or lightweight adapters into `so2_cuda_ops`.
