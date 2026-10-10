<!-- SO2CUDA_BENCHMARKS_BEGIN -->
## 性能测试

### 1. SO(2) 张量积算子

边上的 SO(2) 卷积先将输入旋转到边的主轴，做按 |m| 分组的共享线性变换，再旋转回原坐标：

$$y_e=D(R_e)^\top\,\mathcal{L}_W\!\left(D(R_e)x_e\right).$$

`m=0` 使用实线性变换，`m>0` 使用 2×2 复结构；边之间共享权重，算子测试不含径向调制。

SO2CUDA 的默认 `dense_pairs` 路线使用按索引打包的 Wigner sandwich 和分组 GEMM，再将输出回写到特征布局；非 MoE 的 `true_dense_pairs` 使用同一类 pack／scatter 和 `F.linear`。EquiformerV3 原版与我们的纯 PyTorch 实现分别显式旋转、计算线性层和逆旋转，均不包含这套 CUDA pack／scatter。

- SO2CUDA：公开 `dense_pairs` 默认、`dense_pairs(forward_mode="indexed_sandwich_multi_grouped")` 与 `true_dense_pairs` 三个候选；每个配置选取数值正确且前向＋反向最快的接口。本表所选接口为 `so2_cuda_ops.deeptb.true_dense_pairs`，逐配置选择见下方详情。
- 我们自己的纯 PyTorch 实现：按 DeePTB 上游 `SO2_Linear` 写法实现的算子，见 [operator_baselines.py](examples/operator_baselines.py)。
- 朴素 PyTorch SO(2) 基线采用 EquiformerV3 原版：[固定提交](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48) 的 `SO3Rotation` ＋ `SO2Linear`（eager，源码不改）；`fc_m0` 的 bias 置零。
- cuEquivariance 0.12.0：`SO3` irreps 的 `escn_tp_compact` 描述符由 `SegmentedPolynomial` 执行。所选 method 为 `naive`，旋转方式为 `pytorch`；每个配置选取数值正确且前向＋反向最快的组合。

NVIDIA H200，计时前后核验独占，严格 FP32、关闭 TF32；几何量按各实现的格式预先计算。各实现使用各自原生特征布局，布局转换在计时区外；峰值显存只保留当前实现的输入、参数和几何量。每路预热 5 次、计时 20 次。下表为前向＋输入和全部权重反向的 ms 中位数；括号内为该实现时间 ÷ SO2CUDA 时间：大于 1 表示 SO2CUDA 更快，小于 1 表示该实现更快。

- 与 EquiformerV3 原版相比：14 个配置中 SO2CUDA 都更慢，对方用时为 SO2CUDA 的 0.62–0.85 倍。
- 与我们的纯 PyTorch 实现相比：20 个配置中 SO2CUDA 都更快，对方用时为 SO2CUDA 的 3.03–5.21 倍。
- 与 cuEquivariance 相比：20 个配置中 SO2CUDA 更快 10 个、更慢 10 个，对方用时为 SO2CUDA 的 0.90–2.21 倍。

**形状**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|
| ℓ=2, C=32, m≤2 | 50,000 | 1.72（0.77×） | 10.85（4.86×） | 4.04（1.81×） | 2.24 |
| ℓ=2, C=64, m≤2 | 50,000 | 3.15（0.83×） | 19.80（5.21×） | 4.33（1.14×） | 3.80 |
| ℓ=2, C=128, m≤2 | 50,000 | 6.94（0.85×） | 38.94（4.79×） | 7.90（0.97×） | 8.14 |
| ℓ=4, C=32, m≤4 | 50,000 | 4.49（0.68×） | 32.66（4.91×） | 7.68（1.16×） | 6.65 |
| ℓ=4, C=64, m≤4 | 50,000 | 9.89（0.70×） | 63.46（4.49×） | 12.88（0.91×） | 14.12 |
| ℓ=4, C=128, m≤4 | 50,000 | 25.55（0.75×） | 130.30（3.84×） | 30.61（0.90×） | 33.95 |
| ℓ=6, C=32, m≤6 | 50,000 | 10.52（0.64×） | 70.39（4.26×） | 16.14（0.98×） | 16.53 |
| ℓ=6, C=64, m≤6 | 50,000 | 24.43（0.67×） | 140.20（3.85×） | 33.30（0.91×） | 36.40 |
| ℓ=6, C=128, m≤6 | 50,000 | 67.53（0.73×） | 293.64（3.15×） | 83.63（0.90×） | 93.12 |

**规模**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|
| ℓ=6, C=32, m≤6 | 20,000 | 4.67（0.62×） | 31.01（4.14×） | 10.09（1.35×） | 7.49 |
| ℓ=6, C=32, m≤6 | 50,000 | 10.52（0.64×） | 70.39（4.26×） | 16.14（0.98×） | 16.53 |
| ℓ=6, C=32, m≤6 | 130,000 | 26.17（0.63×） | 167.92（4.04×） | 37.43（0.90×） | 41.55 |
| ℓ=6, C=128, m≤6 | 20,000 | 27.42（0.71×） | 120.92（3.15×） | 35.14（0.91×） | 38.41 |
| ℓ=6, C=128, m≤6 | 50,000 | 67.53（0.73×） | 293.64（3.15×） | 83.63（0.90×） | 93.12 |
| ℓ=6, C=128, m≤6 | 130,000 | 174.99（0.73×） | 727.12（3.03×） | 216.02（0.90×） | 240.27 |

**一般 irreps**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|
| V 形通道 | 20,000 | N/A | 30.73（4.03×） | 11.40（1.49×） | 7.63 |
| V 形通道 | 50,000 | N/A | 69.74（4.09×） | 19.61（1.15×） | 17.05 |
| V 形通道 | 130,000 | N/A | 167.45（3.91×） | 47.92（1.12×） | 42.86 |
| 输入／输出 irreps 不同 | 20,000 | N/A | 8.98（3.27×） | 6.05（2.21×） | 2.74 |
| 输入／输出 irreps 不同 | 50,000 | N/A | 17.22（4.79×） | 6.18（1.72×） | 3.60 |
| 输入／输出 irreps 不同 | 130,000 | N/A | 38.86（5.07×） | 9.92（1.29×） | 7.66 |

**截断 m**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|
| ℓ=6, C=128, m≤2 | 50,000 | 41.39（0.78×） | 184.70（3.47×） | 49.86（0.94×） | 53.28 |

<details>
<summary>前向计时、四分位、峰值显存与 cuEquivariance 选择</summary>

| 配置 | 有向边 | 实现 | 前向 ms（Q1–Q3） | 前向＋反向 ms（Q1–Q3） | 前向＋反向峰值 GiB |
|---|---:|---|---:|---:|---:|
| ℓ=2, C=32, m≤2 | 50,000 | EquiformerV3 原版 | 0.57（0.57–0.58） | 1.72（1.71–1.73） | 0.56 |
| ℓ=2, C=32, m≤2 | 50,000 | 我们的纯 PyTorch | 1.72（1.71–1.72） | 10.85（10.84–10.87） | 0.56 |
| ℓ=2, C=32, m≤2 | 50,000 | cuEquivariance | 1.38（1.37–1.40） | 4.04（4.01–4.07） | 0.48 |
| ℓ=2, C=32, m≤2 | 50,000 | SO2CUDA | 0.93（0.91–0.95） | 2.24（2.23–2.25） | 0.45 |
| ℓ=2, C=64, m≤2 | 50,000 | EquiformerV3 原版 | 1.06（1.05–1.06） | 3.15（3.15–3.15） | 1.02 |
| ℓ=2, C=64, m≤2 | 50,000 | 我们的纯 PyTorch | 2.68（2.68–2.69） | 19.80（19.79–19.83） | 1.06 |
| ℓ=2, C=64, m≤2 | 50,000 | cuEquivariance | 1.46（1.45–1.48） | 4.33（4.31–4.35） | 0.89 |
| ℓ=2, C=64, m≤2 | 50,000 | SO2CUDA | 1.54（1.53–1.54） | 3.80（3.80–3.81） | 0.83 |
| ℓ=2, C=128, m≤2 | 50,000 | EquiformerV3 原版 | 2.30（2.30–2.30） | 6.94（6.93–6.95） | 1.95 |
| ℓ=2, C=128, m≤2 | 50,000 | 我们的纯 PyTorch | 5.02（5.01–5.03） | 38.94（38.92–38.95） | 2.05 |
| ℓ=2, C=128, m≤2 | 50,000 | cuEquivariance | 2.49（2.49–2.49） | 7.90（7.89–7.91） | 1.70 |
| ℓ=2, C=128, m≤2 | 50,000 | SO2CUDA | 3.12（3.11–3.12） | 8.14（8.13–8.14） | 1.58 |
| ℓ=4, C=32, m≤4 | 50,000 | EquiformerV3 原版 | 1.25（1.24–1.25） | 4.49（4.49–4.49） | 1.48 |
| ℓ=4, C=32, m≤4 | 50,000 | 我们的纯 PyTorch | 3.91（3.89–3.92） | 32.66（32.64–32.69） | 1.45 |
| ℓ=4, C=32, m≤4 | 50,000 | cuEquivariance | 2.38（2.35–2.39） | 7.68（7.67–7.72） | 1.34 |
| ℓ=4, C=32, m≤4 | 50,000 | SO2CUDA | 2.15（2.15–2.15） | 6.65（6.64–6.70） | 1.24 |
| ℓ=4, C=64, m≤4 | 50,000 | EquiformerV3 原版 | 2.73（2.73–2.74） | 9.89（9.88–9.90） | 2.66 |
| ℓ=4, C=64, m≤4 | 50,000 | 我们的纯 PyTorch | 7.10（7.09–7.12） | 63.46（63.44–63.49） | 2.80 |
| ℓ=4, C=64, m≤4 | 50,000 | cuEquivariance | 3.14（3.13–3.14） | 12.88（12.87–12.89） | 2.50 |
| ℓ=4, C=64, m≤4 | 50,000 | SO2CUDA | 4.51（4.51–4.52） | 14.12（14.11–14.13） | 2.37 |
| ℓ=4, C=128, m≤4 | 50,000 | EquiformerV3 原版 | 7.47（7.41–7.55） | 25.55（25.53–25.62） | 5.02 |
| ℓ=4, C=128, m≤4 | 50,000 | 我们的纯 PyTorch | 15.33（15.32–15.34） | 130.30（130.27–130.33） | 5.51 |
| ℓ=4, C=128, m≤4 | 50,000 | cuEquivariance | 7.98（7.97–7.98） | 30.61（30.60–30.63） | 4.83 |
| ℓ=4, C=128, m≤4 | 50,000 | SO2CUDA | 10.87（10.86–10.89） | 33.95（33.93–34.05） | 4.61 |
| ℓ=6, C=32, m≤6 | 50,000 | EquiformerV3 原版 | 2.78（2.78–2.78） | 10.52（10.52–10.53） | 3.24 |
| ℓ=6, C=32, m≤6 | 50,000 | 我们的纯 PyTorch | 7.32（7.31–7.33） | 70.39（70.35–70.41） | 2.79 |
| ℓ=6, C=32, m≤6 | 50,000 | cuEquivariance | 3.60（3.59–3.62） | 16.14（16.12–16.17） | 2.80 |
| ℓ=6, C=32, m≤6 | 50,000 | SO2CUDA | 4.56（4.55–4.56） | 16.53（16.53–16.54） | 2.58 |
| ℓ=6, C=64, m≤6 | 50,000 | EquiformerV3 原版 | 6.66（6.60–6.66） | 24.43（24.40–24.44） | 5.53 |
| ℓ=6, C=64, m≤6 | 50,000 | 我们的纯 PyTorch | 14.53（14.52–14.54） | 140.20（140.18–140.23） | 5.43 |
| ℓ=6, C=64, m≤6 | 50,000 | cuEquivariance | 7.42（7.40–7.43） | 33.30（33.27–33.34） | 5.11 |
| ℓ=6, C=64, m≤6 | 50,000 | SO2CUDA | 10.44（10.43–10.45） | 36.40（36.39–36.42） | 4.93 |
| ℓ=6, C=128, m≤6 | 50,000 | EquiformerV3 原版 | 19.74（19.67–19.84） | 67.53（66.36–69.48） | 10.10 |
| ℓ=6, C=128, m≤6 | 50,000 | 我们的纯 PyTorch | 34.12（33.99–34.15） | 293.64（293.62–293.68） | 10.72 |
| ℓ=6, C=128, m≤6 | 50,000 | cuEquivariance | 20.67（20.65–20.70） | 83.63（83.52–84.30） | 9.74 |
| ℓ=6, C=128, m≤6 | 50,000 | SO2CUDA | 30.12（30.10–30.21） | 93.12（93.04–93.20） | 9.64 |
| ℓ=6, C=32, m≤6 | 20,000 | EquiformerV3 原版 | 1.26（1.26–1.27） | 4.67（4.66–4.68） | 1.33 |
| ℓ=6, C=32, m≤6 | 20,000 | 我们的纯 PyTorch | 4.09（4.07–4.11） | 31.01（30.98–31.05） | 1.15 |
| ℓ=6, C=32, m≤6 | 20,000 | cuEquivariance | 3.14（3.13–3.15） | 10.09（10.07–10.11） | 1.16 |
| ℓ=6, C=32, m≤6 | 20,000 | SO2CUDA | 2.27（2.26–2.29） | 7.49（7.49–7.51） | 1.07 |
| ℓ=6, C=32, m≤6 | 130,000 | EquiformerV3 原版 | 6.94（6.94–6.96） | 26.17（26.16–26.22） | 8.32 |
| ℓ=6, C=32, m≤6 | 130,000 | 我们的纯 PyTorch | 16.92（16.91–16.93） | 167.92（167.89–167.95） | 7.15 |
| ℓ=6, C=32, m≤6 | 130,000 | cuEquivariance | 7.96（7.88–8.02） | 37.43（37.38–37.46） | 7.18 |
| ℓ=6, C=32, m≤6 | 130,000 | SO2CUDA | 11.37（11.36–11.38） | 41.55（41.50–41.62） | 6.61 |
| ℓ=6, C=128, m≤2 | 50,000 | EquiformerV3 原版 | 13.51（13.43–13.58） | 41.39（41.13–43.87） | 7.90 |
| ℓ=6, C=128, m≤2 | 50,000 | 我们的纯 PyTorch | 24.63（24.57–24.65） | 184.70（184.66–184.78） | 10.25 |
| ℓ=6, C=128, m≤2 | 50,000 | cuEquivariance | 15.34（15.28–15.39） | 49.86（49.83–51.17） | 8.67 |
| ℓ=6, C=128, m≤2 | 50,000 | SO2CUDA | 18.55（18.54–18.76） | 53.28（53.20–53.88） | 7.26 |
| ℓ=6, C=128, m≤6 | 20,000 | EquiformerV3 原版 | 8.05（8.00–8.07） | 27.42（27.41–27.44） | 4.09 |
| ℓ=6, C=128, m≤6 | 20,000 | 我们的纯 PyTorch | 14.68（14.67–14.68） | 120.92（120.89–120.94） | 4.34 |
| ℓ=6, C=128, m≤6 | 20,000 | cuEquivariance | 8.95（8.94–8.95） | 35.14（35.12–35.24） | 3.97 |
| ℓ=6, C=128, m≤6 | 20,000 | SO2CUDA | 12.66（12.65–12.68） | 38.41（38.40–38.43） | 3.90 |
| ℓ=6, C=128, m≤6 | 130,000 | EquiformerV3 原版 | 51.10（50.83–53.97） | 174.99（173.82–177.59） | 26.14 |
| ℓ=6, C=128, m≤6 | 130,000 | 我们的纯 PyTorch | 87.98（87.27–90.18） | 727.12（726.97–727.23） | 27.76 |
| ℓ=6, C=128, m≤6 | 130,000 | cuEquivariance | 54.69（54.25–55.54） | 216.02（215.47–216.60） | 25.10 |
| ℓ=6, C=128, m≤6 | 130,000 | SO2CUDA | 78.09（77.77–79.49） | 240.27（240.19–240.49） | 24.94 |
| 输入／输出 irreps 不同 | 130,000 | 我们的纯 PyTorch | 5.33（5.32–5.34） | 38.86（38.85–38.87） | 2.12 |
| 输入／输出 irreps 不同 | 130,000 | cuEquivariance | 3.36（3.36–3.51） | 9.92（9.90–9.93） | 1.93 |
| 输入／输出 irreps 不同 | 130,000 | SO2CUDA | 2.76（2.75–2.84） | 7.66（7.65–7.67） | 1.71 |
| 输入／输出 irreps 不同 | 20,000 | 我们的纯 PyTorch | 1.90（1.89–1.91） | 8.98（8.94–9.00） | 0.38 |
| 输入／输出 irreps 不同 | 20,000 | cuEquivariance | 2.10（2.08–2.11） | 6.05（6.04–6.08） | 0.35 |
| 输入／输出 irreps 不同 | 20,000 | SO2CUDA | 1.09（1.08–1.10） | 2.74（2.73–2.76） | 0.32 |
| 输入／输出 irreps 不同 | 50,000 | 我们的纯 PyTorch | 2.63（2.61–2.64） | 17.22（17.21–17.23） | 0.85 |
| 输入／输出 irreps 不同 | 50,000 | cuEquivariance | 2.28（2.27–2.30） | 6.18（6.16–6.19） | 0.78 |
| 输入／输出 irreps 不同 | 50,000 | SO2CUDA | 1.33（1.32–1.34） | 3.60（3.58–3.64） | 0.70 |
| V 形通道 | 130,000 | 我们的纯 PyTorch | 17.60（17.58–17.61） | 167.45（167.43–167.48） | 7.14 |
| V 形通道 | 130,000 | cuEquivariance | 11.93（11.93–11.94） | 47.92（47.91–47.93） | 6.29 |
| V 形通道 | 130,000 | SO2CUDA | 11.85（11.84–11.86） | 42.86（42.85–42.87） | 6.41 |
| V 形通道 | 20,000 | 我们的纯 PyTorch | 4.13（4.12–4.16） | 30.73（30.71–30.75） | 1.15 |
| V 形通道 | 20,000 | cuEquivariance | 3.74（3.71–3.75） | 11.40（11.39–11.42） | 1.02 |
| V 形通道 | 20,000 | SO2CUDA | 2.31（2.29–2.31） | 7.63（7.63–7.64） | 1.04 |
| V 形通道 | 50,000 | 我们的纯 PyTorch | 7.50（7.49–7.51） | 69.74（69.73–69.76） | 2.79 |
| V 形通道 | 50,000 | cuEquivariance | 4.74（4.74–4.75） | 19.61（19.59–19.63） | 2.46 |
| V 形通道 | 50,000 | SO2CUDA | 4.79（4.79–4.80） | 17.05（17.05–17.06） | 2.50 |

| 配置 | 有向边 | cuEquivariance 描述符 | method | 旋转 | 版本 |
|---|---:|---|---|---|---|
| ℓ=2, C=32, m≤2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=2, C=64, m≤2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=2, C=128, m≤2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=4, C=32, m≤4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=4, C=64, m≤4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=4, C=128, m≤4 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=32, m≤6 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=64, m≤6 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=128, m≤6 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=32, m≤6 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=32, m≤6 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=128, m≤2 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=128, m≤6 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| ℓ=6, C=128, m≤6 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| 输入／输出 irreps 不同 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| 输入／输出 irreps 不同 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| 输入／输出 irreps 不同 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| V 形通道 | 130,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| V 形通道 | 20,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |
| V 形通道 | 50,000 | `escn_tp_compact` | `naive` | `pytorch` | 0.12.0 |

全部候选的计时、数值结果与报错见 [算子 JSON](docs/benchmarks/OP_SPEED_H200.json)。

</details>

<details>
<summary>显式旋转与 indexed sandwich 的消融</summary>

中间变体用完整 Wigner 矩阵的 `torch.bmm` 完成旋转与逆旋转，正 m 的线性变换使用与默认 `dense_pairs` 路线相同的公开 `grouped_gemm_multi`，m=0 使用 `F.linear`。它通过完整 Wigner 旋转、一次 split 与 cat 组织逐 m 线性层，不调用 indexed sandwich 的 CUDA pack／scatter。

依次比较 EquiformerV3 原版、中间变体与 SO2CUDA 的默认 `dense_pairs` 路线。分组 GEMM 的替换和 indexed sandwich 的打包／回写分别体现在两步差异中；三者的几何存储与布局也不同。峰值是几何准备完成后的计算峰值分配显存，包含预先计算的几何。

| 配置 | 有向边 | 实现 | 前向 ms | 前向＋反向 ms | 前向峰值 GiB | 前向＋反向峰值 GiB |
|---|---:|---|---:|---:|---:|---:|
| ℓ=2, C=128, m≤2 | 50,000 | EquiformerV3 原版 | 2.30 | 6.94 | 1.17 | 1.95 |
| ℓ=2, C=128, m≤2 | 50,000 | 显式旋转＋SO2CUDA GEMM | 2.56 | 14.78 | 1.80 | 1.80 |
| ℓ=2, C=128, m≤2 | 50,000 | SO2CUDA dense_pairs 默认 | 3.41 | 15.59 | 1.72 | 1.79 |
| ℓ=2, C=128, m≤2 | 50,000 | SO2CUDA 所选候选 | 3.12 | 8.14 | 1.51 | 1.58 |
| ℓ=4, C=128, m≤4 | 50,000 | EquiformerV3 原版 | 7.47 | 25.55 | 3.28 | 5.02 |
| ℓ=4, C=128, m≤4 | 50,000 | 显式旋转＋SO2CUDA GEMM | 9.15 | 30.53 | 5.19 | 5.19 |
| ℓ=4, C=128, m≤4 | 50,000 | SO2CUDA dense_pairs 默认 | 13.89 | 37.93 | 5.97 | 6.09 |
| ℓ=4, C=128, m≤4 | 50,000 | SO2CUDA 所选候选 | 10.87 | 33.95 | 4.18 | 4.61 |
| ℓ=6, C=32, m≤6 | 50,000 | EquiformerV3 原版 | 2.78 | 10.52 | 2.42 | 3.24 |
| ℓ=6, C=32, m≤6 | 50,000 | 显式旋转＋SO2CUDA GEMM | 3.25 | 17.12 | 3.01 | 3.01 |
| ℓ=6, C=32, m≤6 | 50,000 | SO2CUDA dense_pairs 默认 | 7.16 | 23.45 | 3.70 | 3.74 |
| ℓ=6, C=32, m≤6 | 50,000 | SO2CUDA 所选候选 | 4.56 | 16.53 | 2.24 | 2.58 |
| ℓ=6, C=128, m≤6 | 50,000 | EquiformerV3 原版 | 19.74 | 67.53 | 6.81 | 10.10 |
| ℓ=6, C=128, m≤6 | 50,000 | 显式旋转＋SO2CUDA GEMM | 24.72 | 73.42 | 10.54 | 10.54 |
| ℓ=6, C=128, m≤6 | 50,000 | SO2CUDA dense_pairs 默认 | 39.64 | 99.63 | 14.10 | 14.27 |
| ℓ=6, C=128, m≤6 | 50,000 | SO2CUDA 所选候选 | 30.12 | 93.12 | 8.26 | 9.64 |
| ℓ=6, C=32, m≤6 | 20,000 | EquiformerV3 原版 | 1.26 | 4.67 | 1.01 | 1.33 |
| ℓ=6, C=32, m≤6 | 20,000 | 显式旋转＋SO2CUDA GEMM | 1.49 | 7.36 | 1.24 | 1.24 |
| ℓ=6, C=32, m≤6 | 20,000 | SO2CUDA dense_pairs 默认 | 3.47 | 10.35 | 1.52 | 1.53 |
| ℓ=6, C=32, m≤6 | 20,000 | SO2CUDA 所选候选 | 2.27 | 7.49 | 0.93 | 1.07 |
| ℓ=6, C=32, m≤6 | 130,000 | EquiformerV3 原版 | 6.94 | 26.17 | 6.19 | 8.32 |
| ℓ=6, C=32, m≤6 | 130,000 | 显式旋转＋SO2CUDA GEMM | 8.06 | 43.30 | 7.74 | 7.74 |
| ℓ=6, C=32, m≤6 | 130,000 | SO2CUDA dense_pairs 默认 | 18.01 | 59.37 | 9.51 | 9.62 |
| ℓ=6, C=32, m≤6 | 130,000 | SO2CUDA 所选候选 | 11.37 | 41.55 | 5.71 | 6.61 |

Q1/Q3、峰值 reserved 显存和等价性见 [算子 JSON](docs/benchmarks/OP_SPEED_H200.json)。

</details>

<details>
<summary>SO2CUDA 公开接口选择</summary>

| 配置 | 有向边 | 所选接口 | forward_mode |
|---|---:|---|---|
| ℓ=2, C=32, m≤2 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=2, C=64, m≤2 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=2, C=128, m≤2 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=4, C=32, m≤4 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=4, C=64, m≤4 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=4, C=128, m≤4 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=32, m≤6 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=64, m≤6 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=128, m≤6 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=32, m≤6 | 20,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=32, m≤6 | 130,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=128, m≤2 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=128, m≤6 | 20,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| ℓ=6, C=128, m≤6 | 130,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| 输入／输出 irreps 不同 | 130,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| 输入／输出 irreps 不同 | 20,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| 输入／输出 irreps 不同 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| V 形通道 | 130,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| V 形通道 | 20,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |
| V 形通道 | 50,000 | `so2_cuda_ops.deeptb.true_dense_pairs` | `接口默认` |

三个候选的全部计时与报错保留在算子 JSON 中。

</details>

输出、输入梯度、映射回共同参数的权重梯度及整体旋转等变检查，均通过 FP32 的绝对误差／相对 L2 联合判据；近零参考量使用绝对误差判据。逐项误差和判据见数值证据 JSON。

各实现的参数映射与计时约定见 [docs/operator-benchmark.md](docs/operator-benchmark.md)；旋转与逐 m 线性的源码核查见 [docs/rotation-implementation-audit.md](docs/rotation-implementation-audit.md)。

一般 irreps 表中的 EquiformerV3 为 N/A：原实现要求各 ℓ 的通道数一致；这些行的时间比相对我们的纯 PyTorch 和 cuEquivariance 报告。OOM 按原配置记录。

复现算子测试（在本仓库根目录）：

```bash
pip install cuequivariance-torch==0.12.0
git clone https://github.com/atomicarchitects/equiformer_v3.git && git -C equiformer_v3 checkout a7300c58df683dc99cb48027d5bfd4c887486c48
python examples/so2_operator_speed_test.py --impl all --lmax 6 --channels 128 --edges 50000 --warmup 5 --iterations 20 --eqv3-root equiformer_v3 --json operator.json
```

### 2. UniTB 模型

<!-- SO2CUDA_MODEL_CONTEXT_BEGIN -->
[examples/deeptb_speed_test.py](examples/deeptb_speed_test.py) 使用标准 `dptb.nn.build.build_model` 入口随机初始化模型，生成合成周期结构、成对的周期边和正确形状的随机 H0。无需数据集或检查点。

两个配置均使用 `model_options.embedding.method="unitb"`：[unitb.json](examples/configs/unitb.json) 采用 UniTB 默认架构；[unitb_dense.json](examples/configs/unitb_dense.json) 设置 `num_experts=1`，由构造器选择无共享专家、top-k=1 与均匀 32 通道。两者均为三层、`lmax=6`，示例截断半径为 `r_max=7.4`。

UniTB 的 PDQ-MoE 使用四个路由专家、一个共享专家、top-k=2、共享基底参数化、V 形通道与 H0 先验 CG 路由，并启用局域 QEq 电荷响应头。示例包含两种元素和 f 轨道，flow time 固定为 0。随机 H0 和合成 overlap 只用于性能测试，不代表物理先验或模型预测精度。

三行命令测加速（在本仓库根目录，已有 CUDA PyTorch 与 Toolkit）：

```bash
pip install 'git+https://github.com/Franklalalala/DeePTB.git@1006-stable'
pip install -e .
python examples/deeptb_speed_test.py --model both --json speed.json
```

默认每批 20,000 条有向边：每个周期胞 125 个原子，每原子 80 个邻居。`--model unitb` / `--model dense` 可只测一种模型。`--edges` 指定目标边数，脚本向上取整到完整周期胞并记录实际边数；默认结构下 20,000 / 50,000 / 130,000 均为精确边数。`--side`、`--neighbors` 和 `--spacing` 可调节结构。默认 CUDA 分配器上限为 20 GiB；显存不足时可减小 `--edges`，或按设备容量增加 `--max-memory-gib`。

**我们的纯 PyTorch 实现**在同一块 GPU 上使用同一模型、参数和输入，关闭 SO2CUDA（`SO2_CUDA_BACKEND=off`），由 [naive_baseline.py](examples/naive_baseline.py) 换成直接的 PyTorch 写法：SO2 张量积照 [DeePTB 上游主分支的 `SO2_Linear`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py)，专家层照 [UMA 的 MoLE](https://github.com/facebookresearch/fairchem/blob/3801dac0cc0458a2f8121259a2ce8b23d4dcc5a1/src/fairchem/core/models/uma/nn/mole.py) 线性公式。UniTB 的每个路由专家只对分给它的边做一次普通 `F.linear`，再按路由系数加权，共享专家单独计算；UniTB-dense 按路由组混合权重，每组做一次普通 `F.linear`。路由器、电荷头和其余层两条路线相同。

**加速路线**使用安装后的默认设置，脚本自动恢复原模块并选择 `SO2_CUDA_BACKEND=auto`。每轮恢复路由缓冲区，不做优化器更新。计时包含模型前向与 loss 反向，排除输入生成、复制、缓冲区恢复和首次 JIT 编译；默认预热 3 次、计时 10 次，报告中位数与 Q1 / Q3。计算固定为严格 FP32，禁用 TF32。

JSON 同时记录峰值显存、实际成功调用的加速入口、推理与训练输出和全部参数梯度的最大绝对差及逐张量相对 L2。若我们的纯 PyTorch 实现调用了 SO2CUDA，或加速路线没有进入对应入口，脚本会报错。共用 GPU 的结果可用 `--shared-gpu` 标记。

“我们的纯 PyTorch”是我们按 DeePTB 上游 `SO2_Linear` 与 UMA MoLE 写法自己实现的（[examples/naive_baseline.py](examples/naive_baseline.py)）。朴素 PyTorch SO(2) 基线采用 EquiformerV3 原版 `SO3Rotation` ＋ `SO2Linear`（eager）；模型层的通道布局不适用时标 N/A。
<!-- SO2CUDA_MODEL_CONTEXT_END -->

NVIDIA H200，严格 FP32、计时前后核验独占；每路预热 3 次、计时 10 次。前向＋反向中位数沿用相同模型与计时口径，我们的纯 PyTorch 与 SO2CUDA 列使用既有测量，新增列单独实测。

| 模型 | 有向边 | EquiformerV3 原版 ms | 我们的纯 PyTorch ms | cuEquivariance ms | SO2CUDA ms | 我们的纯 PyTorch／SO2CUDA |
|---|---:|---:|---:|---:|---:|---:|
| UniTB-dense | 20,000 | N/A | 553.74 | 568.68 | 282.84 | 1.96× |
| UniTB-dense | 50,000 | N/A | 1016.89 | 779.83 | 437.13 | 2.33× |
| UniTB-dense | 130,000 | N/A | 2230.38 | 1525.00 | 887.79 | 2.51× |
| UniTB | 20,000 | N/A | 661.75 | 769.25 | 318.77 | 2.08× |
| UniTB | 50,000 | N/A | 1194.16 | 1237.43 | 525.67 | 2.27× |
| UniTB | 130,000 | N/A | 2608.04 | 2664.64 | 1095.05 | 2.38× |

<details>
<summary>模型级 cuEquivariance 候选与选择</summary>

| 模型 | 有向边 | 描述符 | method | 前向＋反向 ms | 状态 |
|---|---:|---|---|---:|---|
| UniTB-dense | 20,000 | `escn_tp_compact` | `naive` | 568.68 | 所选 |
| UniTB-dense | 20,000 | `escn_tp_compact` | `fused_tp` | 1252.18 | passed |
| UniTB-dense | 50,000 | `escn_tp_compact` | `naive` | 779.83 | 所选 |
| UniTB-dense | 50,000 | `escn_tp_compact` | `fused_tp` | 2455.37 | passed |
| UniTB-dense | 130,000 | `escn_tp_compact` | `naive` | 1525.00 | 所选 |
| UniTB-dense | 130,000 | `escn_tp_compact` | `fused_tp` | 5747.11 | passed |
| UniTB | 20,000 | `escn_tp_compact` | `naive` | 769.25 | 所选 |
| UniTB | 20,000 | `escn_tp_compact` | `fused_tp` | 2347.73 | passed |
| UniTB | 50,000 | `escn_tp_compact` | `naive` | 1237.43 | 所选 |
| UniTB | 50,000 | `escn_tp_compact` | `fused_tp` | 4951.68 | passed |
| UniTB | 130,000 | `escn_tp_compact` | `naive` | 2664.64 | 所选 |
| UniTB | 130,000 | `escn_tp_compact` | `fused_tp` | 11847.21 | passed |

每个边数分别用整模型比较全层统一的 `escn_tp_compact` 的 `naive` 与 `fused_tp`，所有六个 SO(2) 层使用同一组合，选择等价性通过且前向＋反向中位数最小的候选。

</details>

UniTB-dense 的 EquiformerV3 为 N/A：最终边更新与节点更新的 SO(2) 输出按 l=0..6 合并后为 [151, 37, 41, 29, 13, 5, 1] 通道，最终节点更新的输入也不均匀；EquiformerV3 原版要求输入与输出各自跨 l 统一通道数。其额外 m=0 输出接口可容纳前面 dense 层的标量门控，但无法表达最终层，故完整模型列为 N/A，不补零。
UniTB 的 EquiformerV3 为 N/A：UniTB 的 V 形隐藏通道、门控后的实际 SO(2) 输出以及最终目标层均跨 l 不均匀，EquiformerV3 原版无法直接表达，完整模型列为 N/A，不补零。

训练输出、推理输出和全部参数梯度均有限，并通过绝对误差／相对 L2 联合判据；近零梯度使用绝对误差判据。

cuEquivariance 路线以标准子模块替换 SO(2) 算子，路由器、电荷头和其余模型层共享。每次前向共享一次 Wigner 构建，旋转用分块 bmm，再按描述符的原生布局执行线性层。N/A 表示该路线未提供忠实的对应实现；合成模型结果不代表物理精度或真实训练吞吐。

完整计时、四分位与误差见 [模型 JSON](docs/benchmarks/MODEL_SPEED_H200.json)、[算子数值证据](docs/benchmarks/EQUIV_OP_L40S.json) 和 [模型数值证据](docs/benchmarks/EQUIV_MODEL_L40S.json)。
<!-- SO2CUDA_BENCHMARKS_END -->
