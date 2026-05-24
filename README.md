# SO2 CUDA Ops

`so2-cuda-ops` is a small CUDA backend package for SO2 tensor-product style workloads. It provides reusable grouped GEMM, indexed sandwich pack/scatter, and materialized scheduler helpers without depending on DeePTB package paths.

It is not DeePTB. DeePTB remains the model/training package; this repository only owns backend operators and low-level scheduling helpers.

## Install

Local development:

```bash
pip install -e .
```

From GitHub:

```bash
pip install git+https://github.com/Franklalalala/SO2CUDA.git
```

The CUDA extension is JIT-built on first use with `torch.utils.cpp_extension`.

## Requirements

- Python 3.9+
- PyTorch with CUDA support
- NVIDIA CUDA toolkit with `nvcc`
- cuBLAS development libraries
- Optional CUTLASS checkout for experimental CUTLASS/CuTe kernels

Set `CUDA_HOME` or `CUDA_PATH` if PyTorch cannot find the toolkit.

## Minimal API

```python
import torch
import so2_cuda_ops as so2

print(so2.is_available())
print(so2.get_backend_config())

so2.set_backend_config(
    backend="indexed_sandwich",
    min_edges=1024,
    materialized_min_edges=1024,
    gemm_strategy="scheduler",
)

pair = torch.randn(11, 2, 5, device="cuda")
ptr = torch.tensor([0, 7, 14, 22], device="cuda", dtype=torch.long)
weight = torch.randn(3, 4, 5, device="cuda")
out = so2.indexed_sandwich_multi([pair], ptr, [weight])[0]
```

## DeePTB Integration

Install this package next to a DeePTB checkout:

```bash
pip install -e /path/to/so2-cuda-ops
pip install -e /path/to/DeePTB
```

Then use the DeePTB adapter modes exactly as before, for example:

```bash
export DPTB_SO2_M_LINEAR_MODE=indexed_sandwich_materialized_scheduled
export DPTB_SO2_MATERIALIZED_SCHEDULED_GEMM_STRATEGY=scheduler
```

DeePTB keeps the old `DPTB_SO2_*` configuration surface for compatibility, while this package exposes the generic `SO2_CUDA_*` surface for direct users.

## Environment Variables

- `SO2_CUDA_BACKEND`: `auto`, `indexed_sandwich`, or `materialized_scheduler`
- `SO2_CUDA_MIN_EDGES`: edge-count gate for indexed sandwich use
- `SO2_CUDA_MATERIALIZED_MIN_EDGES`: edge-count gate for materialized scheduler use
- `SO2_CUDA_GEMM_STRATEGY`: `scheduler`, `block_dense`, or `grouped`
- `SO2_CUDA_FAST_TF32`: enable TF32 grouped GEMM
- `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`: cuBLAS grouped JIT build directory
- `SO2_CUDA_PACK_SCATTER_BUILD_DIR`: pack/scatter JIT build directory
- `SO2_CUDA_SCHEDULER_BUILD_DIR`: scheduler JIT build directory
- `SO2_CUDA_CUTLASS_ROOT`: optional CUTLASS checkout root

## Tests And Smoke Benchmarks

```bash
python -m pytest tests -q
python -c "import so2_cuda_ops; print(so2_cuda_ops.is_available())"
```

CUDA correctness tests skip automatically when CUDA is unavailable. A minimal benchmark is available in `examples/minimal_so2_tp.py`.
