<!-- SO2CUDA_BENCHMARKS_BEGIN -->
## Performance

### 1. SO(2) tensor product operator

The edge operator rotates features into the edge frame, applies a shared linear map for each |m|, and rotates back:

$$y_e = D(R_e)^{\top}\,\mathcal{L}_W\!\left(D(R_e)\,x_e\right).$$

The m=0 block is real; each positive m uses a complex linear map represented by real weights. Weights are shared across edges; the operator benchmark excludes radial modulation.

**Uniform irreps** have the same number of channels at every l, as in the usual eSCN, EquiformerV2, and EquiformerV3 layouts. **Non-uniform irreps** have different channel counts across l. SO2CUDA and our pure PyTorch implementation support both, including unequal input and output irreps. The original EquiformerV3 layer requires uniform channels on each side; non-uniform cases are N/A without padding. cuEquivariance support depends on the descriptor and method; unsupported combinations retain their reason in the JSON.

- **SO2CUDA:** the public `true_dense_pairs(..., include_m0=True)` interface, with default settings, computes the complete layer.
- **Our pure PyTorch:** our implementation of [DeePTB upstream `SO2_Linear`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py), in [operator_baselines.py](examples/operator_baselines.py). It is authored here from that computation pattern.
- **EquiformerV3:** the original `SO3Rotation` and `SO2Linear` at [commit `a7300c58`](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48), with m=0 bias set to zero. Eager and `torch.compile(dynamic=True)` are measured separately; compilation is outside steady-state timing.
- **cuEquivariance 0.12.0:** `SegmentedPolynomial` with `escn_tp_compact`, method `naive`, and precomputed PyTorch Wigner `bmm` rotation. All 16 descriptor/method/rotation combinations were compared on 12 uniform configurations; this combination was the fastest numerically correct choice in every configuration. The complete scan is retained in [selection evidence](docs/benchmarks/CUEQ_SELECTION_SCAN.json), and the tables use this fixed choice.

Measurements use NVIDIA H200 in strict FP32 with TF32 disabled. The uniform operator tables use SO2CUDA 0.3.2 source at [`6bc05a7b`](https://github.com/Franklalalala/SO2CUDA/tree/6bc05a7b5135efc153dbb75bf7a20b949979adcf); the other measured tables use [`179faaaa`](https://github.com/Franklalalala/SO2CUDA/tree/179faaaabe269464fd7e0ad1a36926253bd08eb1) for the non-uniform operator table; [`367642e7`](https://github.com/Franklalalala/SO2CUDA/tree/367642e703847d1c6ffc7f84ffdf7e84ad1de713) for UniTB-dense, UniTB, UniTB-SLEM, which execute the same kernels as 0.3.2 on these routes. Geometry and feature-layout conversions are prepared before timing in each implementation's native format. Each implementation has 5 warmup iterations and 20 measured iterations. Each complete table comes from one card task. The main tables show forward + backward medians in ms, including input and weight gradients. Ratios are the comparison time divided by SO2CUDA time: above 1 means SO2CUDA is faster; below 1 means the comparison is faster. Peak allocated memory includes that implementation's inputs, weights, geometry, saved activations, gradients, and workspace.

#### 1.1 Uniform irreps

**Shape**

| Configuration | Directed edges | SO2CUDA ms | Our pure PyTorch ms (ratio) | cuEquivariance ms (ratio) | EquiformerV3 ms (ratio) | EquiformerV3 + compile ms (ratio) |
|---|---:|---:|---:|---:|---:|---:|
| lmax=2, C=32, mmax=2 | 50,000 | 1.01 | 10.88 (10.80×) | 3.98 (3.94×) | 1.71 (1.70×) | 1.30 (1.29×) |
| lmax=2, C=64, mmax=2 | 50,000 | 1.44 | 19.74 (13.75×) | 4.36 (3.04×) | 3.16 (2.20×) | 2.30 (1.60×) |
| lmax=2, C=128, mmax=2 | 50,000 | 3.86 | 38.99 (10.11×) | 7.89 (2.04×) | 6.92 (1.79×) | 5.20 (1.35×) |
| lmax=4, C=32, mmax=4 | 50,000 | 1.89 | 32.53 (17.23×) | 7.71 (4.08×) | 4.48 (2.37×) | 2.75 (1.46×) |
| lmax=4, C=64, mmax=4 | 50,000 | 5.06 | 63.31 (12.50×) | 12.85 (2.54×) | 9.89 (1.95×) | 6.55 (1.29×) |
| lmax=4, C=128, mmax=4 | 50,000 | 16.73 | 130.33 (7.79×) | 30.71 (1.84×) | 25.77 (1.54×) | 19.17 (1.15×) |
| lmax=6, C=32, mmax=6 | 50,000 | 4.29 | 70.54 (16.45×) | 16.20 (3.78×) | 10.53 (2.45×) | 6.50 (1.52×) |
| lmax=6, C=64, mmax=6 | 50,000 | 13.18 | 139.82 (10.61×) | 33.29 (2.52×) | 24.43 (1.85×) | 16.77 (1.27×) |
| lmax=6, C=128, mmax=6 | 50,000 | 45.93 | 293.72 (6.40×) | 83.99 (1.83×) | 68.41 (1.49×) | 52.49 (1.14×) |

**Edge count**

| Configuration | Directed edges | SO2CUDA ms | Our pure PyTorch ms (ratio) | cuEquivariance ms (ratio) | EquiformerV3 ms (ratio) | EquiformerV3 + compile ms (ratio) |
|---|---:|---:|---:|---:|---:|---:|
| lmax=6, C=32, mmax=6 | 20,000 | 2.06 | 31.05 (15.06×) | 10.28 (4.99×) | 4.72 (2.29×) | 2.93 (1.42×) |
| lmax=6, C=32, mmax=6 | 50,000 | 4.29 | 70.54 (16.45×) | 16.20 (3.78×) | 10.53 (2.45×) | 6.50 (1.52×) |
| lmax=6, C=32, mmax=6 | 130,000 | 10.86 | 167.75 (15.45×) | 37.54 (3.46×) | 26.23 (2.42×) | 16.50 (1.52×) |
| lmax=6, C=128, mmax=6 | 20,000 | 18.92 | 120.93 (6.39×) | 35.17 (1.86×) | 27.47 (1.45×) | 21.23 (1.12×) |
| lmax=6, C=128, mmax=6 | 50,000 | 45.93 | 293.72 (6.40×) | 83.99 (1.83×) | 68.41 (1.49×) | 52.49 (1.14×) |
| lmax=6, C=128, mmax=6 | 130,000 | 122.57 | 727.66 (5.94×) | 218.58 (1.78×) | 176.86 (1.44×) | 138.65 (1.13×) |

**Truncated m**

| Configuration | Directed edges | SO2CUDA ms | Our pure PyTorch ms (ratio) | cuEquivariance ms (ratio) | EquiformerV3 ms (ratio) | EquiformerV3 + compile ms (ratio) |
|---|---:|---:|---:|---:|---:|---:|
| lmax=6, C=128, mmax=2 | 50,000 | 33.10 | 184.94 (5.59×) | 49.92 (1.51×) | 41.97 (1.27×) | 36.50 (1.10×) |

<details>
<summary>Forward timing, quartiles, peak memory, and cuEquivariance choices</summary>

| Configuration | Directed edges | Implementation | Forward ms (Q1–Q3) | Forward + backward ms (Q1–Q3) | Forward peak GiB | Forward + backward peak GiB |
|---|---:|---|---:|---:|---:|---:|
| lmax=2, C=32, mmax=2 | 50,000 | SO2CUDA | 0.46 (0.46–0.47) | 1.01 (1.00–1.01) | 0.34 | 0.44 |
| lmax=2, C=32, mmax=2 | 50,000 | Our pure PyTorch | 1.71 (1.70–1.72) | 10.88 (10.85–10.91) | 0.46 | 0.56 |
| lmax=2, C=32, mmax=2 | 50,000 | cuEquivariance | 1.36 (1.36–1.38) | 3.98 (3.96–3.99) | 0.35 | 0.48 |
| lmax=2, C=32, mmax=2 | 50,000 | EquiformerV3 | 0.58 (0.57–0.58) | 1.71 (1.71–1.72) | 0.36 | 0.56 |
| lmax=2, C=32, mmax=2 | 50,000 | EquiformerV3 + compile | 0.58 (0.57–0.58) | 1.30 (1.29–1.31) | 0.36 | 0.47 |
| lmax=2, C=64, mmax=2 | 50,000 | SO2CUDA | 0.60 (0.60–0.61) | 1.44 (1.43–1.44) | 0.61 | 0.82 |
| lmax=2, C=64, mmax=2 | 50,000 | Our pure PyTorch | 2.69 (2.68–2.70) | 19.74 (19.73–19.78) | 0.84 | 1.06 |
| lmax=2, C=64, mmax=2 | 50,000 | cuEquivariance | 1.49 (1.46–2.21) | 4.36 (4.34–4.37) | 0.62 | 0.89 |
| lmax=2, C=64, mmax=2 | 50,000 | EquiformerV3 | 1.06 (1.06–1.06) | 3.16 (3.16–3.17) | 0.63 | 1.02 |
| lmax=2, C=64, mmax=2 | 50,000 | EquiformerV3 + compile | 1.05 (1.04–1.06) | 2.30 (2.29–2.31) | 0.62 | 0.84 |
| lmax=2, C=128, mmax=2 | 50,000 | SO2CUDA | 1.49 (1.49–1.49) | 3.86 (3.85–3.87) | 1.14 | 1.58 |
| lmax=2, C=128, mmax=2 | 50,000 | Our pure PyTorch | 5.02 (5.01–5.03) | 38.99 (38.96–39.07) | 1.62 | 2.05 |
| lmax=2, C=128, mmax=2 | 50,000 | cuEquivariance | 2.49 (2.49–2.50) | 7.89 (7.88–7.89) | 1.16 | 1.70 |
| lmax=2, C=128, mmax=2 | 50,000 | EquiformerV3 | 2.31 (2.30–2.31) | 6.92 (6.92–6.93) | 1.17 | 1.95 |
| lmax=2, C=128, mmax=2 | 50,000 | EquiformerV3 + compile | 2.22 (2.22–2.23) | 5.20 (5.19–5.20) | 1.14 | 1.60 |
| lmax=4, C=32, mmax=4 | 50,000 | SO2CUDA | 0.78 (0.77–0.78) | 1.89 (1.88–1.89) | 0.84 | 1.14 |
| lmax=4, C=32, mmax=4 | 50,000 | Our pure PyTorch | 3.91 (3.90–3.93) | 32.53 (32.52–32.55) | 1.04 | 1.45 |
| lmax=4, C=32, mmax=4 | 50,000 | cuEquivariance | 2.38 (2.36–2.39) | 7.71 (7.69–7.76) | 0.93 | 1.34 |
| lmax=4, C=32, mmax=4 | 50,000 | EquiformerV3 | 1.25 (1.24–1.25) | 4.48 (4.47–4.48) | 1.04 | 1.48 |
| lmax=4, C=32, mmax=4 | 50,000 | EquiformerV3 + compile | 1.20 (1.20–1.21) | 2.75 (2.75–2.80) | 1.05 | 1.22 |
| lmax=4, C=64, mmax=4 | 50,000 | SO2CUDA | 1.94 (1.94–1.94) | 5.06 (5.06–5.07) | 1.59 | 2.19 |
| lmax=4, C=64, mmax=4 | 50,000 | Our pure PyTorch | 7.11 (7.10–7.12) | 63.31 (63.25–63.49) | 1.99 | 2.80 |
| lmax=4, C=64, mmax=4 | 50,000 | cuEquivariance | 3.12 (3.12–3.12) | 12.85 (12.84–12.87) | 1.67 | 2.50 |
| lmax=4, C=64, mmax=4 | 50,000 | EquiformerV3 | 2.74 (2.73–2.74) | 9.89 (9.88–9.90) | 1.79 | 2.66 |
| lmax=4, C=64, mmax=4 | 50,000 | EquiformerV3 + compile | 2.59 (2.59–2.60) | 6.55 (6.55–6.56) | 1.80 | 2.15 |
| lmax=4, C=128, mmax=4 | 50,000 | SO2CUDA | 6.04 (5.86–6.13) | 16.73 (16.65–16.83) | 3.09 | 4.29 |
| lmax=4, C=128, mmax=4 | 50,000 | Our pure PyTorch | 15.37 (15.36–15.37) | 130.33 (130.26–130.37) | 3.89 | 5.51 |
| lmax=4, C=128, mmax=4 | 50,000 | cuEquivariance | 7.91 (7.91–7.92) | 30.71 (30.67–30.95) | 3.18 | 4.83 |
| lmax=4, C=128, mmax=4 | 50,000 | EquiformerV3 | 7.54 (7.49–7.54) | 25.77 (25.67–25.87) | 3.28 | 5.02 |
| lmax=4, C=128, mmax=4 | 50,000 | EquiformerV3 + compile | 7.11 (7.06–7.23) | 19.17 (19.15–19.31) | 3.30 | 4.00 |
| lmax=6, C=32, mmax=6 | 50,000 | SO2CUDA | 1.70 (1.70–1.71) | 4.29 (4.28–4.30) | 1.61 | 2.20 |
| lmax=6, C=32, mmax=6 | 50,000 | Our pure PyTorch | 7.32 (7.30–7.33) | 70.54 (70.51–70.59) | 1.86 | 2.79 |
| lmax=6, C=32, mmax=6 | 50,000 | cuEquivariance | 3.62 (3.61–3.65) | 16.20 (16.18–16.22) | 1.97 | 2.80 |
| lmax=6, C=32, mmax=6 | 50,000 | EquiformerV3 | 2.78 (2.77–2.78) | 10.53 (10.52–10.54) | 2.42 | 3.24 |
| lmax=6, C=32, mmax=6 | 50,000 | EquiformerV3 + compile | 2.64 (2.64–2.65) | 6.50 (6.47–6.53) | 2.41 | 2.75 |
| lmax=6, C=64, mmax=6 | 50,000 | SO2CUDA | 4.91 (4.82–4.91) | 13.18 (13.16–13.26) | 3.08 | 4.25 |
| lmax=6, C=64, mmax=6 | 50,000 | Our pure PyTorch | 14.53 (14.52–14.54) | 139.82 (139.78–139.91) | 3.57 | 5.43 |
| lmax=6, C=64, mmax=6 | 50,000 | cuEquivariance | 7.51 (7.43–7.54) | 33.29 (33.25–33.40) | 3.44 | 5.11 |
| lmax=6, C=64, mmax=6 | 50,000 | EquiformerV3 | 6.66 (6.57–6.78) | 24.43 (24.41–24.44) | 3.88 | 5.53 |
| lmax=6, C=64, mmax=6 | 50,000 | EquiformerV3 + compile | 6.29 (6.25–6.39) | 16.77 (16.72–16.94) | 3.87 | 4.55 |
| lmax=6, C=128, mmax=6 | 50,000 | SO2CUDA | 16.29 (16.28–16.39) | 45.93 (45.68–48.15) | 6.02 | 8.40 |
| lmax=6, C=128, mmax=6 | 50,000 | Our pure PyTorch | 34.35 (34.20–34.39) | 293.72 (293.58–294.24) | 7.00 | 10.72 |
| lmax=6, C=128, mmax=6 | 50,000 | cuEquivariance | 20.86 (20.74–20.89) | 83.99 (83.63–85.28) | 6.41 | 9.74 |
| lmax=6, C=128, mmax=6 | 50,000 | EquiformerV3 | 19.94 (19.86–20.03) | 68.41 (66.91–72.91) | 6.81 | 10.10 |
| lmax=6, C=128, mmax=6 | 50,000 | EquiformerV3 + compile | 19.16 (19.10–19.26) | 52.49 (51.99–55.00) | 6.79 | 8.16 |
| lmax=6, C=32, mmax=6 | 20,000 | SO2CUDA | 0.85 (0.84–0.85) | 2.06 (2.06–2.07) | 0.68 | 0.92 |
| lmax=6, C=32, mmax=6 | 20,000 | Our pure PyTorch | 4.12 (4.10–4.40) | 31.05 (31.02–31.09) | 0.78 | 1.15 |
| lmax=6, C=32, mmax=6 | 20,000 | cuEquivariance | 3.18 (3.16–3.19) | 10.28 (10.26–10.29) | 0.83 | 1.16 |
| lmax=6, C=32, mmax=6 | 20,000 | EquiformerV3 | 1.27 (1.27–1.27) | 4.72 (4.71–4.73) | 1.01 | 1.33 |
| lmax=6, C=32, mmax=6 | 20,000 | EquiformerV3 + compile | 1.24 (1.24–1.25) | 2.93 (2.92–2.94) | 1.00 | 1.14 |
| lmax=6, C=32, mmax=6 | 130,000 | SO2CUDA | 4.03 (4.03–4.06) | 10.86 (10.81–11.07) | 4.08 | 5.60 |
| lmax=6, C=32, mmax=6 | 130,000 | Our pure PyTorch | 16.92 (16.91–16.93) | 167.75 (167.62–167.87) | 4.73 | 7.15 |
| lmax=6, C=32, mmax=6 | 130,000 | cuEquivariance | 7.84 (7.82–7.90) | 37.54 (37.44–37.89) | 5.03 | 7.18 |
| lmax=6, C=32, mmax=6 | 130,000 | EquiformerV3 | 6.94 (6.92–7.04) | 26.23 (26.22–26.26) | 6.19 | 8.32 |
| lmax=6, C=32, mmax=6 | 130,000 | EquiformerV3 + compile | 6.49 (6.47–6.54) | 16.50 (16.45–16.54) | 6.17 | 7.05 |
| lmax=6, C=128, mmax=6 | 20,000 | SO2CUDA | 6.80 (6.79–6.86) | 18.92 (18.89–18.94) | 2.47 | 3.44 |
| lmax=6, C=128, mmax=6 | 20,000 | Our pure PyTorch | 14.69 (14.68–14.70) | 120.93 (120.86–121.01) | 2.85 | 4.34 |
| lmax=6, C=128, mmax=6 | 20,000 | cuEquivariance | 8.94 (8.94–8.94) | 35.17 (35.14–35.60) | 2.63 | 3.97 |
| lmax=6, C=128, mmax=6 | 20,000 | EquiformerV3 | 8.21 (8.00–8.26) | 27.47 (27.41–27.50) | 2.77 | 4.09 |
| lmax=6, C=128, mmax=6 | 20,000 | EquiformerV3 + compile | 7.85 (7.81–7.88) | 21.23 (21.18–21.38) | 2.76 | 3.32 |
| lmax=6, C=128, mmax=6 | 130,000 | SO2CUDA | 42.67 (42.37–47.83) | 122.57 (120.13–126.67) | 15.51 | 21.62 |
| lmax=6, C=128, mmax=6 | 130,000 | Our pure PyTorch | 89.41 (88.05–93.40) | 727.66 (727.46–727.86) | 18.09 | 27.76 |
| lmax=6, C=128, mmax=6 | 130,000 | cuEquivariance | 55.46 (55.04–56.77) | 218.58 (217.11–220.34) | 16.47 | 25.10 |
| lmax=6, C=128, mmax=6 | 130,000 | EquiformerV3 | 52.01 (51.66–53.67) | 176.86 (174.62–180.66) | 17.59 | 26.14 |
| lmax=6, C=128, mmax=6 | 130,000 | EquiformerV3 + compile | 50.18 (49.46–51.62) | 138.65 (136.21–141.97) | 17.53 | 21.07 |
| lmax=6, C=128, mmax=2 | 50,000 | SO2CUDA | 11.74 (11.68–11.76) | 33.10 (32.74–35.75) | 5.06 | 6.95 |
| lmax=6, C=128, mmax=2 | 50,000 | Our pure PyTorch | 24.73 (24.70–24.82) | 184.94 (184.89–184.97) | 7.00 | 10.25 |
| lmax=6, C=128, mmax=2 | 50,000 | cuEquivariance | 15.47 (15.41–15.49) | 49.92 (49.82–51.43) | 6.39 | 8.67 |
| lmax=6, C=128, mmax=2 | 50,000 | EquiformerV3 | 13.60 (13.57–13.65) | 41.97 (41.27–45.16) | 5.01 | 7.90 |
| lmax=6, C=128, mmax=2 | 50,000 | EquiformerV3 + compile | 13.20 (13.18–13.23) | 36.50 (35.80–40.44) | 5.08 | 6.59 |

The cuEquivariance column uses the recorded descriptor, method, and rotation for each configuration:

| Configuration | Directed edges | Descriptor | Method | Rotation |
|---|---:|---|---|---|
| lmax=2, C=32, mmax=2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=2, C=64, mmax=2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=2, C=128, mmax=2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=4, C=32, mmax=4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=4, C=64, mmax=4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=4, C=128, mmax=4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=32, mmax=6 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=64, mmax=6 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=128, mmax=6 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=32, mmax=6 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=32, mmax=6 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=128, mmax=6 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=128, mmax=6 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` |
| lmax=6, C=128, mmax=2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |

Measured candidate timings and unsupported combinations are recorded in [operator JSON](docs/benchmarks/OP_SPEED_H200.json). The original 12-configuration full scan is recorded in [selection evidence](docs/benchmarks/CUEQ_SELECTION_SCAN.json).

</details>

#### 1.2 Non-uniform irreps

The configurations below use decreasing channels, UniTB's V-shaped hidden channels, and unequal input/output irreps:

| Configuration | Input irreps | Output irreps |
|---|---|---|
| Decreasing channels, lmax=2 | `128x0e+64x1o+32x2e` | `128x0e+64x1o+32x2e` |
| Decreasing channels, lmax=4 | `128x0e+64x1o+32x2e+16x3o+16x4e` | `128x0e+64x1o+32x2e+16x3o+16x4e` |
| V-shaped channels | `128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e` | `128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e` |
| V-shaped input → different output | `128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e` | `151x0e+37x1o+41x2e+29x3o+13x4e+5x5o+1x6e` |

| Configuration | Directed edges | SO2CUDA ms | Our pure PyTorch ms (ratio) | cuEquivariance ms (ratio) | EquiformerV3 ms (ratio) | EquiformerV3 + compile ms (ratio) |
|---|---:|---:|---:|---:|---:|---:|
| V-shaped channels | 20,000 | 3.35 | 30.86 (9.22×) | 11.49 (3.43×) | N/A | N/A |
| V-shaped channels | 50,000 | 7.78 | 69.78 (8.96×) | 19.65 (2.52×) | N/A | N/A |
| V-shaped channels | 130,000 | 19.66 | 167.38 (8.51×) | 47.91 (2.44×) | N/A | N/A |
| V-shaped input → different output | 20,000 | 3.04 | 24.37 (8.02×) | 11.62 (3.82×) | N/A | N/A |
| V-shaped input → different output | 50,000 | 6.82 | 52.43 (7.69×) | 18.57 (2.72×) | N/A | N/A |
| V-shaped input → different output | 130,000 | 17.04 | 124.03 (7.28×) | 43.07 (2.53×) | N/A | N/A |
| Decreasing channels, lmax=2 | 20,000 | 1.00 | 7.46 (7.45×) | 4.62 (4.61×) | N/A | N/A |
| Decreasing channels, lmax=2 | 50,000 | 2.11 | 15.80 (7.50×) | 5.42 (2.58×) | N/A | N/A |
| Decreasing channels, lmax=2 | 130,000 | 5.09 | 36.00 (7.07×) | 12.16 (2.39×) | N/A | N/A |
| Decreasing channels, lmax=4 | 20,000 | 1.64 | 13.72 (8.36×) | 7.84 (4.78×) | N/A | N/A |
| Decreasing channels, lmax=4 | 50,000 | 3.51 | 28.75 (8.19×) | 9.39 (2.67×) | N/A | N/A |
| Decreasing channels, lmax=4 | 130,000 | 8.69 | 66.57 (7.67×) | 20.39 (2.35×) | N/A | N/A |

<details>
<summary>Forward timing, quartiles, peak memory, and cuEquivariance choices</summary>

| Configuration | Directed edges | Implementation | Forward ms (Q1–Q3) | Forward + backward ms (Q1–Q3) | Forward peak GiB | Forward + backward peak GiB |
|---|---:|---|---:|---:|---:|---:|
| Decreasing channels, lmax=2 | 20,000 | SO2CUDA | 0.46 (0.46–0.46) | 1.00 (1.00–1.01) | 0.24 | 0.32 |
| Decreasing channels, lmax=2 | 20,000 | Our pure PyTorch | 1.43 (1.42–1.43) | 7.46 (7.44–7.49) | 0.31 | 0.40 |
| Decreasing channels, lmax=2 | 20,000 | cuEquivariance | 1.64 (1.62–1.65) | 4.62 (4.59–4.88) | 0.28 | 0.34 |
| Decreasing channels, lmax=2 | 20,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=2 | 20,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=2 | 50,000 | SO2CUDA | 0.95 (0.95–0.96) | 2.11 (2.11–2.11) | 0.52 | 0.70 |
| Decreasing channels, lmax=2 | 50,000 | Our pure PyTorch | 2.25 (2.24–2.27) | 15.80 (15.78–15.89) | 0.67 | 0.90 |
| Decreasing channels, lmax=2 | 50,000 | cuEquivariance | 1.97 (1.96–1.99) | 5.42 (5.40–5.43) | 0.61 | 0.74 |
| Decreasing channels, lmax=2 | 50,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=2 | 50,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=2 | 130,000 | SO2CUDA | 2.27 (2.27–2.27) | 5.09 (5.09–5.10) | 1.24 | 1.71 |
| Decreasing channels, lmax=2 | 130,000 | Our pure PyTorch | 4.71 (4.70–4.71) | 36.00 (35.99–36.01) | 1.64 | 2.23 |
| Decreasing channels, lmax=2 | 130,000 | cuEquivariance | 4.39 (4.39–4.39) | 12.16 (12.15–12.16) | 1.48 | 1.83 |
| Decreasing channels, lmax=2 | 130,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=2 | 130,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=4 | 20,000 | SO2CUDA | 0.76 (0.76–0.76) | 1.64 (1.64–1.64) | 0.35 | 0.46 |
| Decreasing channels, lmax=4 | 20,000 | Our pure PyTorch | 2.42 (2.41–2.44) | 13.72 (13.69–13.73) | 0.42 | 0.57 |
| Decreasing channels, lmax=4 | 20,000 | cuEquivariance | 2.64 (2.62–2.65) | 7.84 (7.82–7.87) | 0.41 | 0.51 |
| Decreasing channels, lmax=4 | 20,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=4 | 20,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=4 | 50,000 | SO2CUDA | 1.60 (1.60–1.61) | 3.51 (3.50–3.52) | 0.78 | 1.05 |
| Decreasing channels, lmax=4 | 50,000 | Our pure PyTorch | 3.76 (3.75–3.77) | 28.75 (28.72–28.78) | 0.95 | 1.33 |
| Decreasing channels, lmax=4 | 50,000 | cuEquivariance | 3.16 (3.15–3.18) | 9.39 (9.37–9.41) | 0.94 | 1.17 |
| Decreasing channels, lmax=4 | 50,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=4 | 50,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=4 | 130,000 | SO2CUDA | 3.95 (3.94–3.95) | 8.69 (8.68–8.70) | 1.93 | 2.64 |
| Decreasing channels, lmax=4 | 130,000 | Our pure PyTorch | 8.02 (8.01–8.03) | 66.57 (66.56–66.59) | 2.38 | 3.37 |
| Decreasing channels, lmax=4 | 130,000 | cuEquivariance | 6.24 (6.24–6.24) | 20.39 (20.38–20.40) | 2.34 | 2.93 |
| Decreasing channels, lmax=4 | 130,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| Decreasing channels, lmax=4 | 130,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| V-shaped channels | 20,000 | SO2CUDA | 1.44 (1.44–1.45) | 3.35 (3.34–3.35) | 0.68 | 0.92 |
| V-shaped channels | 20,000 | Our pure PyTorch | 4.16 (4.15–4.17) | 30.86 (30.80–30.96) | 0.77 | 1.15 |
| V-shaped channels | 20,000 | cuEquivariance | 3.77 (3.75–3.78) | 11.49 (11.48–11.52) | 0.81 | 1.02 |
| V-shaped channels | 20,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| V-shaped channels | 20,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| V-shaped channels | 50,000 | SO2CUDA | 3.33 (3.33–3.33) | 7.78 (7.78–7.80) | 1.61 | 2.20 |
| V-shaped channels | 50,000 | Our pure PyTorch | 7.51 (7.49–7.52) | 69.78 (69.74–69.85) | 1.82 | 2.79 |
| V-shaped channels | 50,000 | cuEquivariance | 4.75 (4.74–4.75) | 19.65 (19.62–19.70) | 1.93 | 2.46 |
| V-shaped channels | 50,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| V-shaped channels | 50,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| V-shaped channels | 130,000 | SO2CUDA | 8.34 (8.33–8.35) | 19.66 (19.65–19.67) | 4.08 | 5.60 |
| V-shaped channels | 130,000 | Our pure PyTorch | 17.60 (17.59–17.64) | 167.38 (167.27–167.40) | 4.64 | 7.14 |
| V-shaped channels | 130,000 | cuEquivariance | 11.94 (11.93–11.95) | 47.91 (47.90–47.98) | 4.91 | 6.29 |
| V-shaped channels | 130,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| V-shaped channels | 130,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| V-shaped input → different output | 20,000 | SO2CUDA | 1.30 (1.29–1.30) | 3.04 (3.04–3.05) | 0.52 | 0.76 |
| V-shaped input → different output | 20,000 | Our pure PyTorch | 3.51 (3.49–3.53) | 24.37 (24.33–24.41) | 0.63 | 0.85 |
| V-shaped input → different output | 20,000 | cuEquivariance | 3.80 (3.78–3.82) | 11.62 (11.61–11.64) | 0.67 | 0.81 |
| V-shaped input → different output | 20,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| V-shaped input → different output | 20,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| V-shaped input → different output | 50,000 | SO2CUDA | 2.94 (2.94–2.95) | 6.82 (6.81–6.83) | 1.21 | 1.80 |
| V-shaped input → different output | 50,000 | Our pure PyTorch | 5.96 (5.96–5.98) | 52.43 (52.40–52.45) | 1.47 | 2.04 |
| V-shaped input → different output | 50,000 | cuEquivariance | 4.69 (4.67–4.69) | 18.57 (18.55–18.60) | 1.59 | 1.94 |
| V-shaped input → different output | 50,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| V-shaped input → different output | 50,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |
| V-shaped input → different output | 130,000 | SO2CUDA | 7.32 (7.31–7.33) | 17.04 (17.03–17.05) | 3.05 | 4.56 |
| V-shaped input → different output | 130,000 | Our pure PyTorch | 13.55 (13.54–13.56) | 124.03 (123.99–124.09) | 3.72 | 5.21 |
| V-shaped input → different output | 130,000 | cuEquivariance | 10.30 (10.29–10.30) | 43.07 (43.06–43.08) | 4.02 | 4.93 |
| V-shaped input → different output | 130,000 | EquiformerV3 | N/A | N/A | N/A | N/A |
| V-shaped input → different output | 130,000 | EquiformerV3 + compile | N/A | N/A | N/A | N/A |

The cuEquivariance column uses the recorded descriptor, method, and rotation for each configuration:

| Configuration | Directed edges | Descriptor | Method | Rotation |
|---|---:|---|---|---|
| Decreasing channels, lmax=2 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` |
| Decreasing channels, lmax=2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| Decreasing channels, lmax=2 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` |
| Decreasing channels, lmax=4 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` |
| Decreasing channels, lmax=4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| Decreasing channels, lmax=4 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` |
| V-shaped channels | 20,000 | `escn_tp_compact` | `naive` | `pytorch` |
| V-shaped channels | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| V-shaped channels | 130,000 | `escn_tp_compact` | `naive` | `pytorch` |
| V-shaped input → different output | 20,000 | `escn_tp_compact` | `naive` | `pytorch` |
| V-shaped input → different output | 50,000 | `escn_tp_compact` | `naive` | `pytorch` |
| V-shaped input → different output | 130,000 | `escn_tp_compact` | `naive` | `pytorch` |

- EquiformerV3 + compile: The original implementation requires equal channels across l on each side; no padding is used.
- EquiformerV3: The original implementation requires equal channels across l on each side; no padding is used.

Measured candidate timings and unsupported combinations are recorded in [operator JSON](docs/benchmarks/OP_SPEED_H200.json). The original 12-configuration full scan is recorded in [selection evidence](docs/benchmarks/CUEQ_SELECTION_SCAN.json).

</details>

Outputs, input gradients, and weight gradients mapped to common canonical parameters are compared pairwise and against an FP64 reference on small inputs. The new non-uniform irreps also undergo a rotation-equivariance check. [Numerical evidence](docs/benchmarks/EQUIV_OP_L40S.json) records the errors and FP32 tolerances.

**Minimal use for interatomic potentials**

The example accepts e3nn irreps such as `32x0e+32x1o+32x2e` or `128x0e+64x1o+32x2e`, and permits different input/output channel counts. Features use e3nn `mul_ir` layout `[edges, irreps.dim]`. Edge vectors determine a constant rotation; geometry gradients are unsupported. Prepare the layout and Wigner data once for an unchanged edge geometry:

```python
import torch
from e3nn import o3
from examples.minimal_so2_tp import prepare_so2_tp, make_weights
from so2_cuda_ops.deeptb import true_dense_pairs

irreps = o3.Irreps("128x0e+64x1o+32x2e")
edge_vectors = torch.randn(512, 3, device="cuda", dtype=torch.float32)
x = torch.randn(512, irreps.dim, device="cuda", dtype=torch.float32, requires_grad=True)
layout, wigner = prepare_so2_tp(irreps, irreps, edge_vectors, m_max=2)
weights = make_weights(irreps, irreps, 2, device=x.device)
parts = None if wigner is None else true_dense_pairs(x, layout, wigner, weights, include_m0=True)
if parts is None:
    raise RuntimeError("Use your PyTorch reference implementation for this input")
y = parts[0]
y.square().mean().backward()
```

`make_weights` creates trainable W0 and `[A_m; B_m]` matrices, shared across edges; a model can supply its own `LinearWeights` instead. These preparation helpers live in [minimal_so2_tp.py](examples/minimal_so2_tp.py), rather than the package API. The CUDA interface uses FP32 tensors on one CUDA device. CPU, other dtypes, autocast, `torch.func`, geometry requiring gradients, and unsupported layouts return `None`, so the caller can use its reference path. Native CUDA execution errors propagate.

Reproduce an operator comparison from the repository root. The environment must already contain CUDA-enabled PyTorch, compatible cuBLAS, and the standard Python dependencies (NumPy, SciPy, SymPy, NetworkX, opt-einsum, tqdm, nvidia-ml-py, and platformdirs). `--no-deps` preserves that environment; measured software versions are recorded in the JSON:

```bash
pip install --no-deps cuequivariance==0.12.0 cuequivariance-torch==0.12.0 cuequivariance-ops-cu12==0.12.0 cuequivariance-ops-torch-cu12==0.12.0
git clone https://github.com/atomicarchitects/equiformer_v3.git && git -C equiformer_v3 checkout a7300c58df683dc99cb48027d5bfd4c887486c48
python examples/so2_operator_speed_test.py --impl naive,so2cuda,eqv3,cueq --include-compile --suite all --eqv3-root equiformer_v3 --warmup 5 --iterations 20 --cueq-choice escn_tp_compact,naive,pytorch --json operator.json
```

### 2. UniTB models on real training batches

UniTB uses PDQ-MoE; UniTB-dense uses a single expert; UniTB-SLEM applies three SO(2) operators per layer. The comparison uses both onsite and hopping production configurations without model changes. Batches contain real crystal structures from our training set, with a limit of 32 structures per batch. Dynamic cost limits can produce smaller batches; published timing tables report the measured mean structure and directed-edge counts for each fixed batch stream.

For each published model and head, all backends use the same structures in the same order, with 4 warmup steps and 12 measured steps on NVIDIA H200. We switch the SO(2) and associated expert-linear execution backend: default SO2CUDA, [our pure PyTorch implementation](examples/naive_baseline.py), or [cuEquivariance](examples/cueq_baseline.py). Parameters, routing, the remaining model, and the loss stay fixed. HybridMuon uses the same fast optimizer path in every case. Published timing tables show the median complete step (forward + backward + optimizer), its ratio to SO2CUDA, and peak allocated memory. Forward + backward timing excluding the optimizer appears in the details.

The DeePTB source is pinned to [commit `8d5a0dc`](https://github.com/Franklalalala/DeePTB/tree/8d5a0dcda30547f83869c292d48fab5df6eec722); full source commits and software versions are recorded in the public JSON.

**UniTB-dense**

| Head | Mean structures | Mean directed edges | SO2CUDA step ms | SO2CUDA peak GiB | Our pure PyTorch step ms (ratio) | Our pure PyTorch peak GiB | cuEquivariance step ms (ratio) | cuEquivariance peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Onsite | 31.67 | 55,893.0 | 537.45 | 38.34 | 5098.19 (9.49×) | 40.93 | 16878.53 (31.40×) | 35.99 |
| Hopping | 31.67 | 55,893.0 | 532.30 | 38.34 | 5143.27 (9.66×) | 40.93 | 16870.78 (31.69×) | 35.99 |

**UniTB**

| Head | Mean structures | Mean directed edges | SO2CUDA step ms | SO2CUDA peak GiB | Our pure PyTorch step ms (ratio) | Our pure PyTorch peak GiB | cuEquivariance step ms (ratio) | cuEquivariance peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Onsite | 31.67 | 55,893.0 | 792.27 | 32.60 | 1636.29 (2.07×) | 61.44 | 1660.71 (2.10×) | 52.51 |
| Hopping | 31.67 | 55,893.0 | 792.81 | 32.60 | 1628.35 (2.05×) | 61.44 | 1670.60 (2.11×) | 52.51 |

**UniTB-SLEM**

| Head | Mean structures | Mean directed edges | SO2CUDA step ms | SO2CUDA peak GiB | Our pure PyTorch step ms (ratio) | Our pure PyTorch peak GiB | cuEquivariance step ms (ratio) | cuEquivariance peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Onsite | 31.67 | 55,893.0 | 978.87 | 46.25 | 2234.06 (2.28×) | 90.83 | 2308.46 (2.36×) | 75.85 |
| Hopping | 31.67 | 55,893.0 | 993.42 | 46.25 | 2242.09 (2.26×) | 90.83 | 2300.22 (2.32×) | 75.85 |

<details>
<summary>Step quartiles and forward + backward timing without the optimizer</summary>

| Model | Head | Backend | Step ms (Q1–Q3) | Forward + backward ms (Q1–Q3) | Peak allocated GiB |
|---|---|---|---:|---:|---:|
| UniTB-dense | Onsite | SO2CUDA | 537.45 (510.39–577.14) | 458.69 (410.19–484.96) | 38.34 |
| UniTB-dense | Onsite | Our pure PyTorch | 5098.19 (4786.04–5379.31) | 4941.61 (4661.47–5138.95) | 40.93 |
| UniTB-dense | Onsite | cuEquivariance | 16878.53 (16137.94–17580.87) | 16530.35 (15759.40–17242.39) | 35.99 |
| UniTB-dense | Hopping | SO2CUDA | 532.30 (510.02–583.85) | 446.24 (415.44–483.28) | 38.34 |
| UniTB-dense | Hopping | Our pure PyTorch | 5143.27 (4782.17–5449.52) | 4960.46 (4658.51–5192.95) | 40.93 |
| UniTB-dense | Hopping | cuEquivariance | 16870.78 (16151.88–17485.80) | 16521.44 (15743.34–17159.80) | 35.99 |
| UniTB | Onsite | SO2CUDA | 792.27 (765.31–868.40) | 676.27 (631.02–707.64) | 32.60 |
| UniTB | Onsite | Our pure PyTorch | 1636.29 (1553.19–1854.00) | 1511.17 (1375.19–1625.85) | 61.44 |
| UniTB | Onsite | cuEquivariance | 1660.71 (1573.78–1873.63) | 1516.10 (1418.92–1651.25) | 52.51 |
| UniTB | Hopping | SO2CUDA | 792.81 (758.48–857.18) | 673.50 (641.98–717.05) | 32.60 |
| UniTB | Hopping | Our pure PyTorch | 1628.35 (1565.97–1859.38) | 1495.24 (1389.44–1626.72) | 61.44 |
| UniTB | Hopping | cuEquivariance | 1670.60 (1571.91–1873.13) | 1525.19 (1425.87–1636.45) | 52.51 |
| UniTB-SLEM | Onsite | SO2CUDA | 978.87 (936.01–1082.99) | 850.24 (794.41–907.70) | 46.25 |
| UniTB-SLEM | Onsite | Our pure PyTorch | 2234.06 (2096.81–2559.62) | 2081.77 (1899.51–2280.35) | 90.83 |
| UniTB-SLEM | Onsite | cuEquivariance | 2308.46 (2147.07–2626.06) | 2132.64 (1988.37–2328.31) | 75.85 |
| UniTB-SLEM | Hopping | SO2CUDA | 993.42 (921.11–1068.99) | 862.02 (803.25–892.49) | 46.25 |
| UniTB-SLEM | Hopping | Our pure PyTorch | 2242.09 (2096.77–2569.35) | 2077.80 (1898.92–2278.60) | 90.83 |
| UniTB-SLEM | Hopping | cuEquivariance | 2300.22 (2157.93–2648.18) | 2141.75 (1965.90–2332.42) | 75.85 |

</details>

EquiformerV3 is N/A for all three complete models because each contains non-uniform SO(2) layers, including UniTB-dense's final output layers.

The pure PyTorch model baseline is our implementation of the DeePTB upstream SO(2) computation and [UMA MoLE linear formula](https://github.com/facebookresearch/fairchem/blob/3801dac0cc0458a2f8121259a2ce8b23d4dcc5a1/src/fairchem/core/models/uma/nn/mole.py). For each published timing table, the first-step loss and parameter gradients are compared on the same batch within FP32 rounding tolerance, and dispatch checks verify that the pure PyTorch and cuEquivariance routes do not call SO2CUDA. See [model measurement records](docs/benchmarks/MODEL_BS32_H200.json) and [model equivalence](docs/benchmarks/EQUIV_MODEL.json).

The training dataset is not public. Synthetic periodic structures provide a runnable comparison of relative timing; they are used only for timing and do not reproduce the real-batch measurements, physical priors, or prediction accuracy:

```bash
pip install 'git+https://github.com/Franklalalala/DeePTB.git@1006-stable'
pip install -e .
python examples/deeptb_speed_test.py --model all --backend both --edges 20000 --json synthetic_so2cuda.json
python examples/deeptb_speed_test.py --model all --backend cueq --edges 20000 --json synthetic_cueq.json
```

The synthetic example compares the same randomly initialized model and inputs, disables SO2CUDA for reference routes, and records timing, peak memory, dispatch, and numerical differences. Reduce `--edges` on smaller GPUs; `--max-memory-gib` sets its allocator limit. Its forward + backward timings exclude optimizer updates.

<!-- SO2CUDA_BENCHMARKS_END -->
