# Design

The package separates backend responsibilities into three layers:

1. Python metadata and segment preparation.
2. JIT extension loading and grouped GEMM dispatch.
3. CUDA/C++ kernels for pack, scatter, epilogue, and materialized scheduling.

Model-specific shape gates, fallback policy, and training configuration stay in the host project. The backend package receives tensors and descriptors; it does not own DeePTB model construction.
