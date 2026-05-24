# Usage

The public API is intentionally small:

- `is_available()`
- `get_backend_config()`
- `set_backend_config(...)`
- `indexed_sandwich_multi(...)`
- `materialized_scheduler(...)`
- `grouped_gemm(...)`
- `grouped_gemm_multi(...)`

Direct callers should prefer `SO2_CUDA_*` environment variables. DeePTB's adapter keeps `DPTB_SO2_*` compatibility.
