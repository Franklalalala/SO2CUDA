# SO2CUDA

SO2CUDA accelerates SO(2) tensor products and expert linear layers with CUDA, including grouped GEMM, rotated feature packing, output scatter, and execution interfaces for DeePTB's UniTB, UniTB-dense, and UniTB-SLEM models. UniTB uses PDQ-MoE expert layers. The Python package is `so2_cuda_ops`; the current version is **0.3.1**.

## Installation

Requirements: Python 3.9 or later, CUDA-enabled PyTorch, a CUDA Toolkit containing `nvcc`, and cuBLAS development libraries. CUDA extensions are compiled with PyTorch JIT on their first use. Ninja is installed with this package. Use a compiler compatible with the CUDA Toolkit, and set `CUDA_HOME` if PyTorch cannot locate it.

```bash
pip install git+https://github.com/Franklalalala/SO2CUDA.git
```

To develop or run examples from a checkout:

```bash
git clone https://github.com/Franklalalala/SO2CUDA.git
cd SO2CUDA
pip install -e '.[dev]'
```

Check import and device availability, then run the tests:

```bash
python -c "import so2_cuda_ops; print('CUDA available:', so2_cuda_ops.is_available())"
python -m pytest tests -q
```

The package imports on systems without CUDA; `is_available()` returns `False` and GPU tests skip. This function reports device availability. A test or example that executes an operator verifies JIT compilation and the local toolchain.

## Using SO2CUDA with DeePTB

SO2CUDA works with DeePTB's `1006-stable` branch. DeePTB can run through its pure PyTorch reference implementation without SO2CUDA installed. With SO2CUDA installed, layers that meet the CUDA FP32, layout, and routing requirements use the accelerated interfaces. CPU, other dtypes, and unsupported calls use the reference implementation; native CUDA execution errors propagate.

DeePTB calls `so2_cuda_ops.deeptb` through `dptb.nn.so2_backend`. UniTB-dense uses `dense_pairs`; UniTB's PDQ-MoE uses `activation_forward` and grouped GEMM; non-MoE dense layers use `true_dense_pairs`. UniTB-SLEM composes these SO(2) operations with radial modulation. The interfaces accept tensors and small data descriptors. SO2CUDA does not import DeePTB; model parameters and checkpoint structure remain managed by DeePTB.

Grouped linear layers can also be called directly:

```python
import torch
from so2_cuda_ops.deeptb import grouped_gemm

x = torch.randn(1024, 32, device="cuda", dtype=torch.float32)
weight = torch.randn(2, 64, 32, device="cuda", dtype=torch.float32)
# Each contiguous group of rows uses its own [out, in] weight.
ptr = torch.tensor([0, 512, 1024], dtype=torch.long)
y = grouped_gemm(x, ptr, weight)  # [1024, 64], supports autograd
```

<!-- SO2CUDA_BENCHMARKS_BEGIN -->
## Performance

The operator and real training-batch tables are awaiting measurement JSON. Generate them with [update_readme_benchmarks.py](examples/update_readme_benchmarks.py); the generator checks matrix coverage, recorded timing statistics, hardware, precision, and numerical evidence before publishing the tables.
<!-- SO2CUDA_BENCHMARKS_END -->

## Build and runtime settings

- `CUDA_HOME`: CUDA Toolkit directory.
- `TORCH_EXTENSIONS_DIR`: PyTorch JIT extension cache directory.
- `SO2_CUDA_PACK_SCATTER_BUILD_DIR`: build directory for feature packing and scatter extensions.
- `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`: build directory for grouped GEMM extensions.
- `SO2_CUDA_BACKEND`: DeePTB backend policy; defaults to `auto`. Set `off` to force the pure PyTorch path.
- `SO2_CUDA_FAST_TF32`: TF32 switch; the benchmarks set it to `0`.

To place all JIT build products in a chosen location, set both operator build directories and `TORCH_EXTENSIONS_DIR`. Use `CC` and `CXX` to choose compatible compilers; `MAX_JOBS` limits parallel compilation. See [usage.md](docs/usage.md) for interface contracts and [minimal_so2_tp.py](examples/minimal_so2_tp.py) for a complete tensor-product example.
