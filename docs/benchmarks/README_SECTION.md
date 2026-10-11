<!-- SO2CUDA_BENCHMARKS_BEGIN -->
## Operator performance

![Operator performance](docs/figures/operator_performance.png)

*One SO(2) tensor product layer. **a–b**, forward + backward time relative to SO2CUDA for uniform and non-uniform irreps; **c**, peak memory relative to SO2CUDA. Large markers are the scenarios S1, its $m_{\max}=2$ variant and S2; light points are the other measured configurations. Above 1, SO2CUDA is faster or uses less memory.*

| Compared with SO2CUDA | Uniform irreps: time | S1: time | S1: peak memory | Non-uniform irreps: time | S2: time | S2: peak memory |
|---|---:|---:|---:|---:|---:|---:|
| Plain PyTorch | 5.59–17.23× | 6.40× | 1.28× | 7.07–9.22× | 8.96× | 1.27× |
| cuEquivariance | 1.51–4.99× | 1.83× | 1.16× | 2.35–4.78× | 2.52× | 1.12× |
| EquiformerV3 | 1.27–2.45× | 1.49× | 1.20× | N/A | N/A | N/A |
| EquiformerV3 + compile | 1.10–1.60× | 1.14× | 0.97× | N/A | N/A | N/A |

Each entry divides an implementation's forward + backward time (median of 20 measured iterations after 5 warmup iterations) or peak allocated memory by SO2CUDA's, on one NVIDIA H200 in strict FP32 with TF32 disabled; times include input and weight gradients. Geometry is precomputed and each implementation starts from its native feature layout, so the timings exclude Wigner construction and the initial layout conversion, while stored Wigner data count towards peak memory. S1 is `128x0e+128x1o+128x2e+128x3o+128x4e+128x5o+128x6e` and S2 is UniTB's V-shaped hidden irreps `128x0e+24x1o+16x2e+16x3o+32x4e+24x5o+48x6e`, both with 50,000 directed edges; the ranges cover 14 uniform and 12 non-uniform configurations. EquiformerV3 + compile uses slightly less peak memory than SO2CUDA on S1 (0.97×) and on 5 other configurations (0.93–0.98×). Every configuration, timing quartiles and source commits are on the [benchmark page](docs/benchmarks.md).

## Crystal training performance

![Crystal training](docs/figures/crystal_training.png)

*One complete training step of UniTB and UniTB-dense on real crystal batches (onsite head). **a**, step time: SO2CUDA in seconds, the other backends relative to it; **b**, peak memory.*

| Model (onsite head) | SO2CUDA step | Plain PyTorch / SO2CUDA | cuEquivariance / SO2CUDA | Peak GiB: SO2CUDA / Plain PyTorch / cuEquivariance |
|---|---:|---:|---:|---:|
| UniTB | 0.79 s | 2.07× | 2.10× | 32.6 / 61.4 / 52.5 |
| UniTB-dense | 0.54 s | 9.49× | 31.40× | 38.3 / 40.9 / 36.0 |

Batches hold real crystal structures from our training set, at most 32 per batch (on average 31.7 structures and 55.9k directed edges), and every backend trains on the same batches in the same order; the table gives the median of 12 measured complete training steps (forward + backward + HybridMuon optimizer) after 4 warmup steps on one NVIDIA H200 in strict FP32 with TF32 disabled. Only the backend of the SO(2) and expert-linear layers changes, and first-step losses and parameter gradients agree within FP32 tolerance. On UniTB-dense, cuEquivariance uses slightly less peak memory than SO2CUDA (36.0 vs 38.3 GiB). EquiformerV3 is not applicable because both models contain non-uniform SO(2) layers. The hopping head, timing quartiles and forward + backward times without the optimizer are on the [benchmark page](docs/benchmarks.md#models-on-real-training-batches).

<!-- SO2CUDA_BENCHMARKS_END -->
