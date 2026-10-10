# Operator and model benchmark contract

The edge operator is an eSCN-style SO(2) convolution:

\[
y_e=D(R_e)^\top\mathcal L_W(D(R_e)x_e).
\]

Weights are shared across edges; the operator test excludes radial modulation. Code stores row-vector features and a rotation from the principal axis to the edge direction. Input features multiply the Wigner matrix on the right; output features multiply its transpose. Geometry is constant and prepared outside timing in each implementation's native format.

## Canonical parameters and feature layouts

Equivalence checks use e3nn `mul_ir`: irrep block, channel, then m. Each implementation is timed in its own native layout. Input and upstream-gradient conversions take place before timing, and outputs remain native until numerical comparison.

For every retained m, channels follow the irrep block order. The common parameters are `W[0]=W0` and `W[m]=cat([A_m,B_m],dim=0)`. Positive-m pairs have order `(-m,+m)`:

\[
y_0=W_0x_0,\qquad y_-=A_mx_- - B_mx_+,\qquad y_+=A_mx_+ + B_mx_-.
\]

The operator test fixes m=0 biases to zero. Canonical parameters are mapped into each implementation's trainable parameters. Timing includes gradients of those native parameters; mapping and gradient pullback belong to setup and equivalence checking.

- **SO2CUDA** uses public `prepare_layout`, `prepare_wigner`, and `true_dense_pairs(..., include_m0=True)`, with `radial_parts=None` and default settings. This computes m=0 and positive-m contributions in one complete layer call. The example's optional routed interfaces are distinct execution entry points, rather than extra README comparison columns.
- **Our pure PyTorch** implementation follows [DeePTB upstream `SO2_Linear`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py): group input irreps by l for `bmm`, select local features by m, call `F.linear`, combine real and imaginary components, and rotate output irreps back. It is implemented in [operator_baselines.py](../examples/operator_baselines.py), rather than imported unchanged from upstream.
- **EquiformerV3** directly imports [commit `a7300c58`](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3). Its original `SO3Rotation` and `SO2Linear` execute `rotate_inv(linear(rotate(x)))` with native `[E,(lmax+1)^2,C]` features. Original pair order is `(+m,-m)` and pair weights are `[A_m;-B_m]`; channels are permuted by l. For truncated m, inverse rotation scales l>mmax by `sqrt((2*l+1)/(2*mmax+1))`; native output-weight rows compensate for this factor. Gradient pullback includes the same sign, permutation, and scale. Eager and `torch.compile(dynamic=True)` use the same source; compilation and its first call are outside steady-state timing. Original input and output channels must each be uniform across l; the two sides may have different channel counts. Non-uniform irreps are N/A without padding.
- **cuEquivariance 0.12.0** uses public `escn_tp` or `escn_tp_compact` descriptors and `SegmentedPolynomial`. `SO3` descriptors express the complete shared SO(2) parameter family tested here; an O3 descriptor adds parity constraints. `escn_tp` uses `ir_mul`; the compact descriptor groups features by m. The adapter reads actual normalized path coefficients and maps each native parameter as `w_p[u,v]=sigma_p*W_m[v,u]/c_p`, then pulls gradients back by `sigma_p/c_p`. The PyTorch rotation choice combines descriptor ordering with precomputed Wigner matrices and `bmm`; the public `Rotation` candidate uses native `ir_mul` with compact m reordering when required. Selected descriptors, methods, and rotation choices are recorded per configuration.

## Uniform and non-uniform irreps

Uniform irreps use the same multiplicity for every l in a contiguous angular range. The shape grid varies angular order and channel count; the edge-count grid varies batch scale; a truncated-m case follows a common EquiformerV2/V3 setting. Non-uniform cases use decreasing channels, UniTB's V-shaped hidden channels, and different input/output channel distributions. See the README and JSON for the actual configurations.

SO2CUDA and our pure PyTorch implementation cover both groups. EquiformerV3 requires uniform multiplicity on each side. cuEquivariance method support depends on descriptor shapes; unsupported candidates keep their reason, and requested-shape out-of-memory results remain OOM.

## Timing and numerical checks

Forward runs without autograd. Forward + backward differentiates the input and every trainable weight. Formal operator timing uses at least five warmups and twenty measured iterations, with CUDA synchronization and median/Q1/Q3 statistics. GPU exclusivity is checked before and after measurement. Computation is strict FP32 with TF32 disabled; cuEquivariance uses `math_dtype=torch.float32`.

Only the current implementation's geometry, layout indices, weights, native input, and upstream gradient stay resident. Canonical setup data are on CPU, and other implementations' tensors and layout-conversion temporaries are released before timing. Peak allocated memory includes resident data, outputs, saved activations, gradients, and workspace; reserved memory additionally reflects allocator behavior.

When no explicit `--cueq-choice` is supplied, the tool compares supported combinations of both descriptors with `naive`, `uniform_1d`, `fused_tp`, and `indexed_linear`, using PyTorch Wigner rotation or the public cuEquivariance `Rotation`. A candidate must pass the numerical checks before its full-operator timing can be selected. The selected route and alternatives are retained in JSON. README ratios are computed from recorded medians rather than copied from reports.

Small-input checks compare forward output, input gradients, and canonical weight gradients pairwise and against an FP64 reference. Global rotation checks rotate both features and edge vectors. FP32 absolute and relative error criteria allow accumulation-order rounding differences and use absolute error near zero.

Each complete operator table is measured in one card task. Uniform tables may share their overlapping configurations within that task; the non-uniform table may use a separate task. Raw private provenance contains the card records; public JSON contains source commits, software versions, configuration, measurements, and sanitized diagnostics.

## Real training batches

UniTB-dense, UniTB, and UniTB-SLEM use unchanged production configurations for onsite and hopping. For each model/head pair, all backends receive the same real crystal structures in the same order. Batching limits the number of structures and uses the production dynamic cost limits; reported means describe the actual measured stream.

The backend adapters switch SO(2) and associated expert-linear execution. [naive_baseline.py](../examples/naive_baseline.py) follows the upstream SO(2) computation and UMA MoLE linear formula. [cueq_baseline.py](../examples/cueq_baseline.py) replaces SO(2) with cuEquivariance while retaining the original parameter objects and checkpoint keys. Radial placement, expert routing, activation order, charge response, remaining model, loss, and the fast HybridMuon optimizer path stay fixed. PDQ-MoE mixes expert outputs using the original gates before activation; shared experts remain separate unless folded by the model's defined parameterization.

The cuEquivariance route builds compact per-l Wigner blocks once per model forward and passes them to all SO(2) layers. Rotation groups features by l for `bmm`; local features use the selected descriptor's native arrangement. Parameter mapping occurs in model forward and is included in model timing.

Model timing uses at least four warmup steps and twelve measured steps. Complete-step timing measures synchronized wall time around the production trainer iteration, including batch preparation, forward/backward, gradient clipping, metrics, and the fast HybridMuon update. The separate forward + backward measurement sums CUDA-event intervals from training-payload construction through each expert's loss backward; it excludes batch preparation, clipping, metrics, and the optimizer. Peak memory describes the complete measured step. First-step loss and parameter gradients are checked on a common batch. Dispatch checks require zero SO2CUDA calls on pure PyTorch and cuEquivariance reference routes. Each model's two-head table comes from one card task.

EquiformerV3 is N/A for complete models: UniTB and UniTB-SLEM have non-uniform hidden channels, and UniTB-dense's final SO(2) output layers are non-uniform. No third-party padding, channel truncation, or source edits are used.

The training dataset is not public. [deeptb_speed_test.py](../examples/deeptb_speed_test.py) supplies synthetic periodic structures for a runnable relative-timing comparison. Its measurements exclude optimizer updates and are distinct from real training-batch results.

## Generate the README

[update_readme_benchmarks.py](../examples/update_readme_benchmarks.py) accepts complete operator and model JSON plus their numerical-equivalence receipts. It checks matrix coverage and m cutoffs, hardware, precision, warmup/sample counts, recorded quartiles, peak memory, and required table-session provenance. Recorded batch counts must reproduce the reported dynamic-batch means. Individual configurations must be complete even when their aggregate is marked complete. Only allowlisted fields enter public JSON.

```bash
python examples/update_readme_benchmarks.py --operator-json operator.json --model-json model_batches.json --equiv-operator-json equiv_operator.json --equiv-model-json equiv_model.json --write-readme
```

Run the same command with `--check` in place of `--write-readme` to verify that the README and public JSON reproduce exactly, without writing files. Results live in `docs/benchmarks/`.
