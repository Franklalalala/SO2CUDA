# Usage

The tensor-only integration entry point is `so2_cuda_ops.deeptb`. The caller manages model construction, radial networks, expert parameterization, and checkpoints. These interfaces accept tensors and small dataclasses, and SO2CUDA does not import DeePTB.

| Interface | Purpose |
|---|---|
| `prepare_layout`, `PairLayout` | Build integer maps for each m from `(l, multiplicity, first_feature)` entries; reuse on the same layer and device |
| `prepare_wigner`, `WignerData` | Accept dense Wigner matrices or compact per-l blocks; geometry is constant |
| `dense_pairs`, `DenseRouting` | Return ordered m>0 contributions; with `include_m0=True`, return the complete layer output |
| `true_dense_pairs` | Accept non-routed `LinearWeights`; with `include_m0=True`, include the m=0 bias and radial block |
| `activation_forward`, `ActivationRouting`, `LinearWeights` | Pack features, execute grouped expert linears, and scatter activation-space top-k outputs |
| `grouped_gemm`, `grouped_gemm_multi` | Differentiable segmented matrix multiplication for one problem or multiple m blocks |
| `permute_rows` | Apply a bijective row permutation; backward reads the inverse permutation |

## Features, geometry, and weights

The CUDA SO(2) interfaces use FP32. Features have shape `[N, Din]`; floating tensors must share a CUDA device. Ordinary non-contiguous inputs are accepted and made contiguous where required. Layout entries identify the angular order, multiplicity, and first feature index for each irrep. e3nn `mul_ir` orders features by irrep block, channel, and m. Both uniform and non-uniform multiplicities can be described, and input and output irreps can differ.

Wigner data can be dense `[N, D, D]` matrices or a tuple of `[N, 2*l+1, 2*l+1]` blocks, one for each l. `prepare_wigner` packs the data into the representation expected by the kernels. Geometry must not require gradients. Reuse prepared layout and Wigner descriptors while the layer and edge directions remain unchanged.

For a complete non-MoE operator, `true_dense_pairs` accepts one non-routed `LinearWeights` per m. The m=0 weight is `[Cout_0, Cin_0]`, with an optional bias `[Cout_0]`. Positive-m weights are `[2*Cout_m, Cin_m]`, formed by stacking `[A_m; B_m]`. With local component order `(-m,+m)`, they implement:

\[
y_0=W_0x_0+b_0,\qquad y_-=A_mx_- - B_mx_+,\qquad y_+=A_mx_+ + B_mx_-.
\]

`include_m0=True` returns a sequence containing one tensor: the complete rotated-back layer output. The default `False` leaves m=0 to the caller and returns contributions to add in order. [minimal_so2_tp.py](../examples/minimal_so2_tp.py) prepares e3nn irreps and edge-vector geometry, creates trainable weights, and calls this interface without DeePTB.

## Routing and radial modulation

Grouped linear weights have shape `[G, Dout, Din]`. `ptr` is an int64 prefix pointer of length `G+1` covering input rows. `DenseRouting.ptr` covers flattened `[N,2]` pair rows, with a bijective row permutation when needed. `dense_pairs` indexes `weights` and `radial_parts` by m, including an m=0 placeholder in the default mode. Positive-m weights are `[G, 2*Cout_m, Cin_m]`. With `include_m0=True`, `weights[0]` is `[G, Cout_0, Cin_0]`, and the single returned tensor is the whole layer output.

Radial blocks are `[N,Cin_m]` for input-side (`front=True`) modulation or `[N,Cout_m]` for output-side modulation. The m=0 radial block is also included when `include_m0=True`.

`ActivationRouting.indices` and `values` have shape `[N,K]`. Each slot describes a stable expert ordering as `(order, inverse, ptr_cpu, sorted_expert_ids)`. Routed `LinearWeights.weight` has shape `[experts,Dout,Din]`; non-routed interpolation blocks use `[Dout,Din]`. Set `coefficients_sum_to_one=True` only when the caller has already folded shared experts according to the model definition; otherwise provide the separate shared weights.

## Fallback and errors

SO(2) interfaces return `None` before unsupported native execution for CPU, non-FP32, autocast, `torch.func`, geometry requiring gradients, or unsupported layouts. The caller then uses its reference implementation. DeePTB handles that fallback through its optional backend. Native execution exceptions propagate; they do not silently select another path.

Direct grouped GEMM requires CUDA FP32. Its single-problem interface also supports JVP. The SO(2) interfaces themselves do not support `torch.func` transforms.

Install with `pip install -e .`. `is_available()` reports whether PyTorch detects CUDA. The first actual operator call JIT-compiles the extensions and checks the local toolchain. The package can be imported without CUDA.

## Build settings

Set `SO2_CUDA_PACK_SCATTER_BUILD_DIR`, `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`, and `TORCH_EXTENSIONS_DIR` to choose build and cache locations. Use `CUDA_HOME`, `CC`, and `CXX` for a compatible CUDA Toolkit and compiler, and `MAX_JOBS` to limit concurrent compilation. `SO2_CUDA_FAST_TF32=0` keeps the benchmark FP32 policy. DeePTB accepts `SO2_CUDA_BACKEND=off` to force reference execution.

See [operator-benchmark.md](operator-benchmark.md) for parameter mappings and measurement boundaries. Run `python examples/so2_operator_speed_test.py --help` for grids, optional dependencies, and equivalence-check options.
