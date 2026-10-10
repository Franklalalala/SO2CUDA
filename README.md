# SO2CUDA

SO2CUDA 为 SO(2) 张量积和专家线性层提供 CUDA 加速，包括分组 GEMM、旋转后的张量打包与输出 scatter，以及 DeePTB 的 UniTB / UniTB-dense 执行接口。UniTB 使用 PDQ-MoE 专家层。Python 包名为 `so2_cuda_ops`，当前版本为 **0.3.0**。

## 安装

需要 Python 3.9 或更高版本、支持 CUDA 的 PyTorch、包含 `nvcc` 的 CUDA Toolkit 和 cuBLAS 开发库。首次调用 CUDA 算子时通过 PyTorch JIT 编译；Ninja 随本包安装。编译器与 CUDA Toolkit 应彼此兼容；PyTorch 找不到 Toolkit 时可设置 `CUDA_HOME`。

```bash
pip install git+https://github.com/Franklalalala/SO2CUDA.git
```

在本仓库中开发或运行示例：

```bash
git clone https://github.com/Franklalalala/SO2CUDA.git
cd SO2CUDA
pip install -e '.[dev]'
```

检查导入和 CUDA 设备可用性：

```bash
python -c "import so2_cuda_ops; print('CUDA available:', so2_cuda_ops.is_available())"
python -m pytest tests -q
```

无 CUDA 设备时可以导入本包，`is_available()` 返回 `False`，GPU 测试会跳过。`is_available()` 检查设备可用性；实际 JIT 构建和算子执行由测试或下方模型示例验证。

## 与 DeePTB 配合

SO2CUDA 0.3.0 与 DeePTB 的 `1006-stable` 分支配合使用。DeePTB 不安装 SO2CUDA 也能通过纯 PyTorch 运行；安装后，符合 CUDA FP32、布局与路由条件的层自动调用加速接口。CPU、其他 dtype 或不支持的调用使用参考实现，实际 CUDA 执行错误会直接抛出。

DeePTB 通过 `dptb.nn.so2_backend` 调用 `so2_cuda_ops.deeptb`：UniTB-dense 使用 `dense_pairs`，UniTB 的 PDQ-MoE 使用 `activation_forward` 和分组 GEMM；扩展的非 MoE dense 层使用 `true_dense_pairs`。接口只接受张量与数据描述对象，SO2CUDA 本身不导入 DeePTB。模型参数与检查点结构由 DeePTB 管理。

分组线性层也可以直接使用：

```python
import torch
from so2_cuda_ops.deeptb import grouped_gemm

x = torch.randn(1024, 32, device="cuda", dtype=torch.float32)
weight = torch.randn(2, 64, 32, device="cuda", dtype=torch.float32)
# Each contiguous group of input rows uses its own [out, in] weight.
ptr = torch.tensor([0, 512, 1024], dtype=torch.long)
y = grouped_gemm(x, ptr, weight)  # [1024, 64], supports autograd
```

<!-- SO2CUDA_BENCHMARKS_BEGIN -->
## 性能测试

### 1. SO(2) 张量积算子

边上的 SO(2) 卷积先将输入旋转到边的主轴，做按 |m| 分组的共享线性变换，再旋转回原坐标：

$$y_e=D(R_e)^\top\,\mathcal{L}_W\!\left(D(R_e)x_e\right).$$

`m=0` 使用实线性变换，`m>0` 使用 2×2 复结构；边之间共享权重，算子测试不含径向调制。

EquiformerV3 原版本身就是显式旋转：把系数排列并入稠密 Wigner 矩阵后用 `bmm` 旋转，再做逐 m 线性和逆旋转。SO2CUDA 与它数学相同，差别在 indexed sandwich 的实现：CUDA kernel 以（边，通道）为单位，把输入的各 ℓ 分量旋入每个 m 一块的缓冲；每块做一次 GEMM，m>0 块的 2×2 复结构写成一个实数块权重；再从各块旋回并写出特征。反向由同样的 kernel 与 GEMM 完成。我们的纯 PyTorch 实现按 DeePTB 上游写法逐 l 旋转、按 m 选取特征后做线性层。

- SO2CUDA：通用入口 `true_dense_pairs(..., include_m0=True)`，一次调用算完整层（含 m=0）。其他公开入口（`dense_pairs`、单专家的 `activation_forward`）执行同一套 kernel，各入口逐配置的计时见 [docs/operator-benchmark.md](docs/operator-benchmark.md#各入口计时)。
- 我们自己的纯 PyTorch 实现：按 DeePTB 上游 `SO2_Linear` 写法实现的算子，见 [operator_baselines.py](examples/operator_baselines.py)。
- EquiformerV3 原版：[固定提交](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48) 的 `SO3Rotation` ＋ `SO2Linear`（eager，源码不改），`fc_m0` 的 bias 置零；另列同一实现经 `torch.compile` 后的时间（各 ℓ 通道数一致的 14 个配置）。
- cuEquivariance 0.12.0：`SO3` irreps 的 `escn_tp_compact` 描述符由 `SegmentedPolynomial` 执行，method 为 `naive`，旋转方式为 `pytorch`；这一组合在全部描述符／method／旋转组合中前向＋反向最快且数值正确。

全部配置与实现在同一块 NVIDIA H200 上的同一次会话中测量，计时前后核验独占，严格 FP32、关闭 TF32；几何量按各实现的格式预先计算。各实现使用各自原生特征布局，布局转换在计时区外；峰值显存只保留当前实现的输入、参数和几何量。每路预热 5 次、计时 20 次。下表为前向＋输入和全部权重反向的 ms 中位数；括号内为该实现时间 ÷ SO2CUDA 时间：大于 1 表示 SO2CUDA 更快，小于 1 表示该实现更快。SO2CUDA 列均为 `true_dense_pairs(include_m0=True)`。

- 与 EquiformerV3 原版相比：14 个配置中 SO2CUDA 都更快，对方用时为 SO2CUDA 的 1.21–2.03 倍。
- 与 EquiformerV3 + compile 相比：14 个配置中 SO2CUDA 都更快，对方用时为 SO2CUDA 的 1.05–1.39 倍。
- 与我们的纯 PyTorch 实现相比：20 个配置中 SO2CUDA 都更快，对方用时为 SO2CUDA 的 5.38–14.71 倍。
- 与 cuEquivariance 相比：20 个配置中 SO2CUDA 都更快，对方用时为 SO2CUDA 的 1.45–5.49 倍。

**形状**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | EquiformerV3 + compile ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|---:|
| ℓ=2, C=32, m≤2 | 50,000 | 1.71（1.84×） | 1.28（1.38×） | 10.82（11.62×） | 3.93（4.22×） | 0.93 |
| ℓ=2, C=64, m≤2 | 50,000 | 3.16（1.93×） | 2.28（1.39×） | 19.73（12.07×） | 4.33（2.65×） | 1.64 |
| ℓ=2, C=128, m≤2 | 50,000 | 6.93（1.66×） | 5.20（1.24×） | 38.78（9.27×） | 7.87（1.88×） | 4.18 |
| ℓ=4, C=32, m≤4 | 50,000 | 4.47（2.03×） | 2.74（1.24×） | 32.47（14.71×） | 7.62（3.45×） | 2.21 |
| ℓ=4, C=64, m≤4 | 50,000 | 9.88（1.76×） | 6.54（1.17×） | 63.32（11.31×） | 12.88（2.30×） | 5.60 |
| ℓ=4, C=128, m≤4 | 50,000 | 25.65（1.48×） | 19.06（1.10×） | 130.07（7.48×） | 30.65（1.76×） | 17.39 |
| ℓ=6, C=32, m≤6 | 50,000 | 10.51（1.76×） | 6.53（1.09×） | 70.19（11.75×） | 16.13（2.70×） | 5.97 |
| ℓ=6, C=64, m≤6 | 50,000 | 24.43（1.59×） | 16.69（1.09×） | 139.81（9.13×） | 33.32（2.18×） | 15.32 |
| ℓ=6, C=128, m≤6 | 50,000 | 67.62（1.42×） | 52.28（1.10×） | 293.35（6.16×） | 83.90（1.76×） | 47.64 |

**规模**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | EquiformerV3 + compile ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|---:|
| ℓ=6, C=32, m≤6 | 20,000 | 4.66（1.74×） | 2.92（1.09×） | 30.86（11.54×） | 10.05（3.76×） | 2.67 |
| ℓ=6, C=32, m≤6 | 50,000 | 10.51（1.76×） | 6.53（1.09×） | 70.19（11.75×） | 16.13（2.70×） | 5.97 |
| ℓ=6, C=32, m≤6 | 130,000 | 26.20（1.76×） | 16.41（1.10×） | 167.47（11.27×） | 37.47（2.52×） | 14.86 |
| ℓ=6, C=128, m≤6 | 20,000 | 27.40（1.39×） | 21.16（1.07×） | 120.59（6.10×） | 35.14（1.78×） | 19.76 |
| ℓ=6, C=128, m≤6 | 50,000 | 67.62（1.42×） | 52.28（1.10×） | 293.35（6.16×） | 83.90（1.76×） | 47.64 |
| ℓ=6, C=128, m≤6 | 130,000 | 177.86（1.40×） | 137.86（1.09×） | 727.30（5.72×） | 217.99（1.72×） | 127.06 |

**一般 irreps**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | EquiformerV3 + compile ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|---:|
| V 形通道 | 20,000 | N/A | N/A | 30.78（8.32×） | 11.40（3.08×） | 3.70 |
| V 形通道 | 50,000 | N/A | N/A | 69.77（7.99×） | 19.55（2.24×） | 8.73 |
| V 形通道 | 130,000 | N/A | N/A | 167.12（7.55×） | 47.89（2.16×） | 22.14 |
| 输入／输出 irreps 不同 | 20,000 | N/A | N/A | 8.64（7.83×） | 6.06（5.49×） | 1.10 |
| 输入／输出 irreps 不同 | 50,000 | N/A | N/A | 17.23（12.53×） | 6.24（4.54×） | 1.38 |
| 输入／输出 irreps 不同 | 130,000 | N/A | N/A | 38.87（12.43×） | 9.94（3.18×） | 3.13 |

**截断 m**

| 配置 | 有向边 | EquiformerV3 原版 ms（时间比） | EquiformerV3 + compile ms（时间比） | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|---:|
| ℓ=6, C=128, m≤2 | 50,000 | 41.52（1.21×） | 36.19（1.05×） | 184.72（5.38×） | 49.92（1.45×） | 34.35 |

<details>
<summary>前向计时、四分位与峰值显存</summary>

| 配置 | 有向边 | 实现 | 前向 ms（Q1–Q3） | 前向＋反向 ms（Q1–Q3） | 前向＋反向峰值 GiB |
|---|---:|---|---:|---:|---:|
| ℓ=2, C=32, m≤2 | 50,000 | EquiformerV3 原版 | 0.58（0.57–0.58） | 1.71（1.71–1.73） | 0.56 |
| ℓ=2, C=32, m≤2 | 50,000 | EquiformerV3 + compile | 0.58（0.57–0.58） | 1.28（1.28–1.29） | 0.47 |
| ℓ=2, C=32, m≤2 | 50,000 | 我们的纯 PyTorch | 1.71（1.70–1.71） | 10.82（10.81–10.87） | 0.56 |
| ℓ=2, C=32, m≤2 | 50,000 | cuEquivariance | 1.36（1.34–1.37） | 3.93（3.93–3.95） | 0.48 |
| ℓ=2, C=32, m≤2 | 50,000 | SO2CUDA | 0.42（0.41–0.42） | 0.93（0.93–0.94） | 0.44 |
| ℓ=2, C=64, m≤2 | 50,000 | EquiformerV3 原版 | 1.06（1.06–1.06） | 3.16（3.15–3.16） | 1.02 |
| ℓ=2, C=64, m≤2 | 50,000 | EquiformerV3 + compile | 1.04（1.04–1.05） | 2.28（2.27–2.28） | 0.84 |
| ℓ=2, C=64, m≤2 | 50,000 | 我们的纯 PyTorch | 2.68（2.67–2.69） | 19.73（19.71–19.74） | 1.06 |
| ℓ=2, C=64, m≤2 | 50,000 | cuEquivariance | 1.46（1.45–1.47） | 4.33（4.31–4.35） | 0.89 |
| ℓ=2, C=64, m≤2 | 50,000 | SO2CUDA | 0.71（0.71–0.71） | 1.64（1.63–1.64） | 0.82 |
| ℓ=2, C=128, m≤2 | 50,000 | EquiformerV3 原版 | 2.30（2.30–2.31） | 6.93（6.92–6.93） | 1.95 |
| ℓ=2, C=128, m≤2 | 50,000 | EquiformerV3 + compile | 2.22（2.22–2.22） | 5.20（5.20–5.21） | 1.60 |
| ℓ=2, C=128, m≤2 | 50,000 | 我们的纯 PyTorch | 5.01（5.01–5.02） | 38.78（38.77–38.82） | 2.05 |
| ℓ=2, C=128, m≤2 | 50,000 | cuEquivariance | 2.49（2.49–2.49） | 7.87（7.87–7.88） | 1.70 |
| ℓ=2, C=128, m≤2 | 50,000 | SO2CUDA | 1.66（1.66–1.66） | 4.18（4.18–4.19） | 1.58 |
| ℓ=4, C=32, m≤4 | 50,000 | EquiformerV3 原版 | 1.24（1.24–1.25） | 4.47（4.47–4.47） | 1.48 |
| ℓ=4, C=32, m≤4 | 50,000 | EquiformerV3 + compile | 1.20（1.20–1.21） | 2.74（2.74–2.74） | 1.22 |
| ℓ=4, C=32, m≤4 | 50,000 | 我们的纯 PyTorch | 3.89（3.88–3.91） | 32.47（32.44–32.48） | 1.45 |
| ℓ=4, C=32, m≤4 | 50,000 | cuEquivariance | 2.35（2.33–2.36） | 7.62（7.60–7.64） | 1.34 |
| ℓ=4, C=32, m≤4 | 50,000 | SO2CUDA | 0.93（0.93–0.94） | 2.21（2.20–2.21） | 1.14 |
| ℓ=4, C=64, m≤4 | 50,000 | EquiformerV3 原版 | 2.74（2.74–2.74） | 9.88（9.88–9.88） | 2.66 |
| ℓ=4, C=64, m≤4 | 50,000 | EquiformerV3 + compile | 2.60（2.60–2.61） | 6.54（6.53–6.55） | 2.15 |
| ℓ=4, C=64, m≤4 | 50,000 | 我们的纯 PyTorch | 7.10（7.09–7.11） | 63.32（63.31–63.38） | 2.80 |
| ℓ=4, C=64, m≤4 | 50,000 | cuEquivariance | 3.14（3.13–3.14） | 12.88（12.88–12.90） | 2.50 |
| ℓ=4, C=64, m≤4 | 50,000 | SO2CUDA | 2.22（2.21–2.22） | 5.60（5.59–5.61） | 2.19 |
| ℓ=4, C=128, m≤4 | 50,000 | EquiformerV3 原版 | 7.44（7.42–7.57） | 25.65（25.62–25.71） | 5.02 |
| ℓ=4, C=128, m≤4 | 50,000 | EquiformerV3 + compile | 7.10（7.09–7.14） | 19.06（19.05–19.09） | 4.00 |
| ℓ=4, C=128, m≤4 | 50,000 | 我们的纯 PyTorch | 15.34（15.33–15.37） | 130.07（130.04–130.14） | 5.51 |
| ℓ=4, C=128, m≤4 | 50,000 | cuEquivariance | 8.04（7.99–8.05） | 30.65（30.63–30.68） | 4.83 |
| ℓ=4, C=128, m≤4 | 50,000 | SO2CUDA | 6.39（6.26–6.41） | 17.39（17.37–17.40） | 4.29 |
| ℓ=6, C=32, m≤6 | 50,000 | EquiformerV3 原版 | 2.78（2.78–2.78） | 10.51（10.50–10.52） | 3.24 |
| ℓ=6, C=32, m≤6 | 50,000 | EquiformerV3 + compile | 2.63（2.63–2.64） | 6.53（6.49–6.54） | 2.75 |
| ℓ=6, C=32, m≤6 | 50,000 | 我们的纯 PyTorch | 7.31（7.29–7.32） | 70.19（70.16–70.22） | 2.79 |
| ℓ=6, C=32, m≤6 | 50,000 | cuEquivariance | 3.59（3.58–3.61） | 16.13（16.11–16.14） | 2.80 |
| ℓ=6, C=32, m≤6 | 50,000 | SO2CUDA | 2.54（2.54–2.54） | 5.97（5.97–5.99） | 2.20 |
| ℓ=6, C=64, m≤6 | 50,000 | EquiformerV3 原版 | 6.69（6.66–6.74） | 24.43（24.43–24.46） | 5.53 |
| ℓ=6, C=64, m≤6 | 50,000 | EquiformerV3 + compile | 6.32（6.27–6.34） | 16.69（16.61–16.82） | 4.55 |
| ℓ=6, C=64, m≤6 | 50,000 | 我们的纯 PyTorch | 14.51（14.50–14.53） | 139.81（139.80–139.85） | 5.43 |
| ℓ=6, C=64, m≤6 | 50,000 | cuEquivariance | 7.47（7.45–7.51） | 33.32（33.27–33.40） | 5.11 |
| ℓ=6, C=64, m≤6 | 50,000 | SO2CUDA | 5.89（5.89–5.90） | 15.32（15.30–15.34） | 4.25 |
| ℓ=6, C=128, m≤6 | 50,000 | EquiformerV3 原版 | 19.90（19.89–19.99） | 67.62（66.83–71.44） | 10.10 |
| ℓ=6, C=128, m≤6 | 50,000 | EquiformerV3 + compile | 19.04（19.01–19.08） | 52.28（51.77–54.90） | 8.16 |
| ℓ=6, C=128, m≤6 | 50,000 | 我们的纯 PyTorch | 34.13（34.11–34.18） | 293.35（293.29–293.55） | 10.72 |
| ℓ=6, C=128, m≤6 | 50,000 | cuEquivariance | 20.91（20.90–20.93） | 83.90（83.61–84.58） | 9.74 |
| ℓ=6, C=128, m≤6 | 50,000 | SO2CUDA | 17.14（17.02–17.24） | 47.64（47.21–48.46） | 8.40 |
| ℓ=6, C=128, m≤2 | 50,000 | EquiformerV3 原版 | 13.57（13.50–13.58） | 41.52（41.26–43.94） | 7.90 |
| ℓ=6, C=128, m≤2 | 50,000 | EquiformerV3 + compile | 13.10（13.08–13.13） | 36.19（35.65–39.22） | 6.59 |
| ℓ=6, C=128, m≤2 | 50,000 | 我们的纯 PyTorch | 24.67（24.66–24.76） | 184.72（184.71–184.78） | 10.25 |
| ℓ=6, C=128, m≤2 | 50,000 | cuEquivariance | 15.43（15.37–15.48） | 49.92（49.78–51.29） | 8.67 |
| ℓ=6, C=128, m≤2 | 50,000 | SO2CUDA | 12.43（12.38–12.48） | 34.35（33.91–37.84） | 6.95 |
| ℓ=6, C=32, m≤6 | 20,000 | EquiformerV3 原版 | 1.27（1.27–1.27） | 4.66（4.65–4.68） | 1.33 |
| ℓ=6, C=32, m≤6 | 20,000 | EquiformerV3 + compile | 1.24（1.23–1.24） | 2.92（2.91–2.92） | 1.14 |
| ℓ=6, C=32, m≤6 | 20,000 | 我们的纯 PyTorch | 4.05（4.05–4.08） | 30.86（30.85–30.88） | 1.15 |
| ℓ=6, C=32, m≤6 | 20,000 | cuEquivariance | 3.11（3.10–3.13） | 10.05（10.03–10.06） | 1.16 |
| ℓ=6, C=32, m≤6 | 20,000 | SO2CUDA | 1.14（1.13–1.14） | 2.67（2.67–2.68） | 0.92 |
| ℓ=6, C=32, m≤6 | 130,000 | EquiformerV3 原版 | 6.95（6.94–6.99） | 26.20（26.19–26.22） | 8.32 |
| ℓ=6, C=32, m≤6 | 130,000 | EquiformerV3 + compile | 6.51（6.44–6.54） | 16.41（16.34–16.46） | 7.05 |
| ℓ=6, C=32, m≤6 | 130,000 | 我们的纯 PyTorch | 16.91（16.90–16.93） | 167.47（167.44–167.51） | 7.15 |
| ℓ=6, C=32, m≤6 | 130,000 | cuEquivariance | 7.89（7.84–7.96） | 37.47（37.44–37.49） | 7.18 |
| ℓ=6, C=32, m≤6 | 130,000 | SO2CUDA | 6.22（6.22–6.23） | 14.86（14.85–14.86） | 5.60 |
| ℓ=6, C=128, m≤6 | 20,000 | EquiformerV3 原版 | 8.13（8.01–8.20） | 27.40（27.38–27.42） | 4.09 |
| ℓ=6, C=128, m≤6 | 20,000 | EquiformerV3 + compile | 7.84（7.75–7.94） | 21.16（21.05–21.19） | 3.32 |
| ℓ=6, C=128, m≤6 | 20,000 | 我们的纯 PyTorch | 14.67（14.65–14.69） | 120.59（120.56–120.62） | 4.34 |
| ℓ=6, C=128, m≤6 | 20,000 | cuEquivariance | 8.95（8.94–8.95） | 35.14（35.13–35.38） | 3.97 |
| ℓ=6, C=128, m≤6 | 20,000 | SO2CUDA | 7.15（7.14–7.26） | 19.76（19.67–19.80） | 3.44 |
| ℓ=6, C=128, m≤6 | 130,000 | EquiformerV3 原版 | 51.81（51.57–54.03） | 177.86（175.77–179.02） | 26.14 |
| ℓ=6, C=128, m≤6 | 130,000 | EquiformerV3 + compile | 49.51（49.38–52.40） | 137.86（135.52–141.15） | 21.07 |
| ℓ=6, C=128, m≤6 | 130,000 | 我们的纯 PyTorch | 88.81（87.55–93.31） | 727.30（727.08–727.46） | 27.76 |
| ℓ=6, C=128, m≤6 | 130,000 | cuEquivariance | 55.01（54.60–56.19） | 217.99（217.53–218.52） | 25.10 |
| ℓ=6, C=128, m≤6 | 130,000 | SO2CUDA | 44.96（44.60–48.38） | 127.06（124.95–130.84） | 21.62 |
| V 形通道 | 20,000 | 我们的纯 PyTorch | 4.16（4.15–4.25） | 30.78（30.77–30.80） | 1.15 |
| V 形通道 | 20,000 | cuEquivariance | 3.72（3.71–3.74） | 11.40（11.38–11.41） | 1.02 |
| V 形通道 | 20,000 | SO2CUDA | 1.62（1.61–1.62） | 3.70（3.69–3.70） | 0.92 |
| V 形通道 | 50,000 | 我们的纯 PyTorch | 7.50（7.50–7.51） | 69.77（69.73–69.85） | 2.79 |
| V 形通道 | 50,000 | cuEquivariance | 4.75（4.75–4.75） | 19.55（19.53–19.57） | 2.46 |
| V 形通道 | 50,000 | SO2CUDA | 3.80（3.79–3.80） | 8.73（8.72–8.74） | 2.20 |
| V 形通道 | 130,000 | 我们的纯 PyTorch | 17.60（17.58–17.61） | 167.12（167.09–167.21） | 7.14 |
| V 形通道 | 130,000 | cuEquivariance | 11.93（11.93–11.94） | 47.89（47.88–47.90） | 6.29 |
| V 形通道 | 130,000 | SO2CUDA | 9.59（9.59–9.60） | 22.14（22.13–22.16） | 5.60 |
| 输入／输出 irreps 不同 | 20,000 | 我们的纯 PyTorch | 1.90（1.88–1.91） | 8.64（8.64–8.66） | 0.38 |
| 输入／输出 irreps 不同 | 20,000 | cuEquivariance | 2.08（2.06–2.08） | 6.06（6.04–6.07） | 0.35 |
| 输入／输出 irreps 不同 | 20,000 | SO2CUDA | 0.48（0.47–0.48） | 1.10（1.09–1.11） | 0.29 |
| 输入／输出 irreps 不同 | 50,000 | 我们的纯 PyTorch | 2.62（2.61–2.63） | 17.23（17.21–17.24） | 0.85 |
| 输入／输出 irreps 不同 | 50,000 | cuEquivariance | 2.29（2.27–2.30） | 6.24（6.23–6.25） | 0.78 |
| 输入／输出 irreps 不同 | 50,000 | SO2CUDA | 0.65（0.65–0.66） | 1.38（1.36–1.66） | 0.62 |
| 输入／输出 irreps 不同 | 130,000 | 我们的纯 PyTorch | 5.33（5.32–5.34） | 38.87（38.86–38.89） | 2.12 |
| 输入／输出 irreps 不同 | 130,000 | cuEquivariance | 3.37（3.36–3.37） | 9.94（9.92–9.96） | 1.93 |
| 输入／输出 irreps 不同 | 130,000 | SO2CUDA | 1.48（1.48–1.49） | 3.13（3.12–3.13） | 1.51 |

全部实现与 SO2CUDA 各公开入口的计时、数值核对与报错见 [算子 JSON](docs/benchmarks/OP_SPEED_H200.json)。

</details>

每个配置计时前，各实现（含 SO2CUDA 的各公开入口）先在 128 条边上与 FP64 参考比较输出、输入梯度和映射回共同参数的权重梯度，并做整体旋转等变检查，均通过 FP32 的绝对误差／相对 L2 联合判据；近零参考量使用绝对误差判据。逐项误差和判据见 [算子数值证据](docs/benchmarks/EQUIV_OP_H200.json)。

各实现的参数映射与计时约定见 [docs/operator-benchmark.md](docs/operator-benchmark.md)。

一般 irreps 表中 EquiformerV3 的两列为 N/A：原实现要求各 ℓ 的通道数一致；这些行的时间比相对我们的纯 PyTorch 和 cuEquivariance 报告。

复现算子测试（在本仓库根目录）：

```bash
pip install cuequivariance-torch==0.12.0
git clone https://github.com/atomicarchitects/equiformer_v3.git && git -C equiformer_v3 checkout a7300c58df683dc99cb48027d5bfd4c887486c48
python examples/so2_operator_speed_test.py --impl naive,so2cuda,eqv3,cueq --include-compile --cueq-choice escn_tp_compact,naive,pytorch --lmax 6 --channels 128 --edges 50000 --warmup 5 --iterations 20 --eqv3-root equiformer_v3 --json operator.json
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

与上节算子测试在同一块 NVIDIA H200 上的同一次会话中测量，严格 FP32，每次计时迭代前后核验独占；每路先做一次不计时的前向＋反向，再预热 3 次、计时 10 次。下表为前向＋反向 ms 中位数，括号内为该实现时间 ÷ SO2CUDA 时间。cuEquivariance 列在全部六个 SO(2) 层统一使用 `escn_tp_compact` 描述符与 `naive` method，每次前向共享一次 Wigner 构建。

| 模型 | 有向边 | EquiformerV3 原版 ms | 我们的纯 PyTorch ms（时间比） | cuEquivariance ms（时间比） | SO2CUDA ms |
|---|---:|---:|---:|---:|---:|
| UniTB-dense | 20,000 | N/A | 552.84（2.21×） | 557.79（2.23×） | 250.54 |
| UniTB-dense | 50,000 | N/A | 1015.85（2.82×） | 773.33（2.14×） | 360.75 |
| UniTB-dense | 130,000 | N/A | 2230.04（3.22×） | 1524.84（2.20×） | 693.52 |
| UniTB | 20,000 | N/A | 659.03（2.07×） | 768.89（2.41×） | 318.50 |
| UniTB | 50,000 | N/A | 1190.57（2.28×） | 1234.09（2.36×） | 523.09 |
| UniTB | 130,000 | N/A | 2608.06（2.38×） | 2660.13（2.43×） | 1094.53 |

<details>
<summary>前向计时、四分位与峰值显存</summary>

| 模型 | 有向边 | 实现 | 前向 ms（Q1–Q3） | 前向＋反向 ms（Q1–Q3） | 峰值 GiB |
|---|---:|---|---:|---:|---:|
| UniTB-dense | 20,000 | 我们的纯 PyTorch | 146.75（146.16–148.07） | 552.84（550.60–557.87） | 10.72 |
| UniTB-dense | 20,000 | cuEquivariance | 177.00（176.55–178.21） | 557.79（555.11–563.16） | 9.34 |
| UniTB-dense | 20,000 | SO2CUDA | 96.28（96.07–96.64） | 250.54（249.76–253.34） | 8.95 |
| UniTB-dense | 50,000 | 我们的纯 PyTorch | 202.15（201.51–205.56） | 1015.85（1014.23–1018.50） | 26.57 |
| UniTB-dense | 50,000 | cuEquivariance | 205.28（205.02–207.78） | 773.33（772.08–778.31） | 23.06 |
| UniTB-dense | 50,000 | SO2CUDA | 131.10（130.80–131.97） | 360.75（360.54–361.59） | 22.04 |
| UniTB-dense | 130,000 | 我们的纯 PyTorch | 388.67（388.22–390.03） | 2230.04（2228.79–2231.29） | 68.83 |
| UniTB-dense | 130,000 | cuEquivariance | 322.07（321.58–324.54） | 1524.84（1522.62–1528.25） | 59.65 |
| UniTB-dense | 130,000 | SO2CUDA | 252.61（252.38–252.67） | 693.52（693.47–699.83） | 56.97 |
| UniTB | 20,000 | 我们的纯 PyTorch | 194.64（193.92–200.89） | 659.03（658.43–667.28） | 17.04 |
| UniTB | 20,000 | cuEquivariance | 236.69（235.79–245.77） | 768.89（763.35–786.20） | 14.54 |
| UniTB | 20,000 | SO2CUDA | 126.39（125.77–126.84） | 318.50（317.64–318.93） | 11.41 |
| UniTB | 50,000 | 我们的纯 PyTorch | 286.52（286.25–286.79） | 1190.57（1189.47–1198.61） | 42.34 |
| UniTB | 50,000 | cuEquivariance | 296.12（294.06–302.51） | 1234.09（1226.95–1235.87） | 36.03 |
| UniTB | 50,000 | SO2CUDA | 198.68（198.24–199.08） | 523.09（522.41–524.35） | 28.23 |
| UniTB | 130,000 | 我们的纯 PyTorch | 572.28（572.05–572.71） | 2608.06（2607.20–2608.53） | 109.82 |
| UniTB | 130,000 | cuEquivariance | 541.10（540.57–545.21） | 2660.13（2656.13–2664.00） | 93.34 |
| UniTB | 130,000 | SO2CUDA | 416.61（416.24–416.98） | 1094.53（1094.32–1096.27） | 73.08 |

</details>

UniTB-dense 的 EquiformerV3 为 N/A：最终边更新与节点更新的 SO(2) 输出按 l=0..6 合并后为 [151, 37, 41, 29, 13, 5, 1] 通道，最终节点更新的输入也不均匀；EquiformerV3 原版要求输入与输出各自跨 l 统一通道数。其额外 m=0 输出接口可容纳前面 dense 层的标量门控，但无法表达最终层，故完整模型列为 N/A，不补零。
UniTB 的 EquiformerV3 为 N/A：UniTB 的 V 形隐藏通道、门控后的实际 SO(2) 输出以及最终目标层均跨 l 不均匀，EquiformerV3 原版无法直接表达，完整模型列为 N/A，不补零。

SO2CUDA 与 cuEquivariance 两条路线的训练输出、推理输出和全部参数梯度均与我们的纯 PyTorch 实现比较，全部有限并通过绝对误差／相对 L2 联合判据；近零梯度使用绝对误差判据。

cuEquivariance 路线以标准子模块替换 SO(2) 算子，路由器、电荷头和其余模型层共享；旋转用分块 bmm，再按描述符的原生布局执行线性层。N/A 表示该路线未提供忠实的对应实现；合成模型结果不代表物理精度或真实训练吞吐。

完整计时、四分位与误差见 [模型 JSON](docs/benchmarks/MODEL_SPEED_H200.json) 和 [模型数值证据](docs/benchmarks/EQUIV_MODEL.json)。
<!-- SO2CUDA_BENCHMARKS_END -->

## 构建与运行设置

- `CUDA_HOME`：CUDA Toolkit 目录。
- `TORCH_EXTENSIONS_DIR`：PyTorch JIT 扩展缓存目录。
- `SO2_CUDA_PACK_SCATTER_BUILD_DIR`：张量打包与 scatter 扩展构建目录。
- `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`：分组 GEMM 扩展构建目录。
- `SO2_CUDA_BACKEND`：DeePTB 后端策略，默认 `auto`，`off` 强制纯 PyTorch。
- `SO2_CUDA_FAST_TF32`：TF32 开关；上述示例固定为 `0`。

需要把全部 JIT 产物放到指定位置时，同时设置两个算子构建目录与 `TORCH_EXTENSIONS_DIR`。`CC` / `CXX` 可选择与 Toolkit 兼容的编译器，`MAX_JOBS` 控制并行编译数。更多接口约定见 [docs/usage.md](docs/usage.md)。

通用张量积算子的最小示例见 [examples/minimal_so2_tp.py](examples/minimal_so2_tp.py)。
