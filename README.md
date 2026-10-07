# SO2CUDA

SO2CUDA 为 SO(2) 张量积和专家线性层提供 CUDA 加速，包括分组 GEMM、旋转后的张量打包与输出 scatter，以及 DeePTB UniTB 的 dense / X1 执行接口。Python 包名为 `so2_cuda_ops`，当前版本为 **0.2.0**。

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

SO2CUDA 0.2.0 与 DeePTB 的 `1006-stable` 分支配合使用。DeePTB 不安装 SO2CUDA 也能通过纯 PyTorch 运行；安装后，符合 CUDA FP32、布局与路由条件的层自动调用加速接口。CPU、其他 dtype 或不支持的调用使用参考实现，实际 CUDA 执行错误会直接抛出。

在本仓库根目录安装两个包：

```bash
pip install -e .
pip install 'git+https://github.com/Franklalalala/DeePTB.git@1006-stable'
```

DeePTB 通过 `dptb.nn.so2_backend` 调用 `so2_cuda_ops.deeptb`：UniTB-dense 使用 `dense_pairs`，UniTB-X1 使用 `activation_forward` 和分组 GEMM；扩展的非 MoE dense 层使用 `true_dense_pairs`。`grouped_gemm`、`grouped_gemm_multi`、`permute_rows` 与布局准备接口也可单独调用。接口只接受张量与数据描述对象，SO2CUDA 本身不导入 DeePTB。模型参数与检查点结构由 DeePTB 管理。

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

## dense 与 X1 加速测试

[examples/deeptb_speed_test.py](examples/deeptb_speed_test.py) 使用标准 `dptb.nn.build.build_model` 入口随机初始化模型，自动生成周期结构、成对的周期边和正确形状的随机 H0。无需数据集或检查点。

两个配置均使用 `model_options.embedding.method="unitb"`。X1 的 embedding 采用 UniTB 默认架构；dense 只需增加 `num_experts=1`，相应的无共享专家、top-k=1 与均匀 32 通道由构造器选择。示例统一设置 `r_max=7.4` 以固定合成结构的截断半径。

UniTB-X1 使用 H₀-routed shared-basis mixture of experts (PDQ-MoE)：四个路由专家、一个共享专家、top-k=2、共享低秩专家参数、V 形通道与先验 CG 路由。两种配置均为三层、`lmax=6`；X1 的独立 `shift_head` 配置还启用局域 QEq 响应头。示例采用两种元素的精简基组，包含 f 轨道以覆盖最高阶张量积。flow time 固定为 0；随机 H0 和合成 overlap 只用于性能测试，不代表物理先验或模型预测精度。

三行命令测加速（在本仓库根目录，已有 CUDA PyTorch 与 Toolkit）：

```bash
pip install 'git+https://github.com/Franklalalala/DeePTB.git@1006-stable'
pip install -e .
python examples/deeptb_speed_test.py --model both --forward-mode indexed_sandwich_multi --json speed.json
```

默认每批 20,000 条有向边：每个周期胞 125 个原子，每原子 80 个邻居。`--model dense` / `--model x1` 可只测一种模型。`--edges` 指定目标边数，脚本向上取整到完整周期胞并记录实际边数；默认结构下 20,000 / 50,000 / 130,000 均为精确边数。`--side`、`--neighbors` 和 `--spacing` 可调节结构。默认 CUDA 分配器上限为 20 GiB，遇到显存不足时减小边数；大边数测试可按设备容量增加 `--max-memory-gib`：

```bash
python examples/deeptb_speed_test.py --model both --edges 10000 --iterations 5 --json speed.json
```

脚本在同一模型、输入和参数上分别运行 SO2CUDA 关闭与开启路线。关闭时设置 `SO2_CUDA_BACKEND=off`，使用 `staged` 张量积和 `split_loop` 专家线性层；开启时设置 `auto`，dense 使用 `dense_pairs`，X1 使用激活空间 fused-P0 与 cuBLAS 分组 GEMM。每轮恢复路由缓冲区，不做优化器更新。计时包含模型前向与 loss 反向，排除输入生成、复制、缓冲区恢复和首次 JIT 编译；默认先预热三轮，再记录十轮的中位数与 Q1 / Q3（包含端点的四分位插值）。

结果包括前向、反向与合计 ms/iter、加速比、峰值分配与保留显存、实际成功调用的加速入口，以及推理输出、训练输出和参数梯度的最大绝对差与相对 L2 差。CUDA 模式会核验对应的加速入口确实执行，未进入加速路线会报错。计算固定为严格 FP32，禁用 TF32；参考与 CUDA 路线的求和顺序可能造成 FP32 舍入差，JSON 会给出实测数值。峰值分配包含模型、输入、训练中间张量与梯度；保留显存还包含分配器缓存。

独占 H200 实测中，dense 使用 `indexed_sandwich_multi` 的前向＋反向为参考路线的 1.84–2.36 倍速度，因此上方三行命令显式选择该模式。示例的自动选择仍为 H200 `scalar`、其他 GPU `indexed_sandwich_multi`；库的默认值保持不变。正常模型运行可在启动前设置：

```bash
export DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=indexed_sandwich_multi
```

示例会根据命令行参数设置同一环境变量，运行示例时同时使用 `--forward-mode`：

```bash
python examples/deeptb_speed_test.py --model dense --forward-mode indexed_sandwich_multi
```

共用 GPU 的结果可能有计时噪声；用 `--shared-gpu` 将这一情况记入 JSON。CPU 或未安装 SO2CUDA 的 DeePTB 环境可以先做纯 PyTorch 小规模检查：

```bash
python examples/deeptb_speed_test.py --model both --device cpu --backend reference \
  --edges 64 --side 2 --neighbors 8 --warmup 0 --iterations 1 --json cpu_smoke.json
```

<!-- FIN_H200_TABLE -->

### H200 实测（2026-10-07）

GPU：NVIDIA H200，逐次核对所用物理 GPU 的进程表，确认独占。严格 FP32，关闭 TF32；随机初始化 UniTB 模型与合成周期结构，每组 SO2CUDA 开／关使用相同参数和输入，无优化器更新。每路预热 3 次、计时 10 次。加速比为关闭／开启 SO2CUDA 的前向＋反向合计中位数；大于 1 表示更快。

| 模型 | 有向边 | scalar 默认 | indexed_sandwich_multi 显式覆盖 |
|---|---:|---:|---:|
| UniTB-dense | 20,000 | 0.716× | 1.839× |
| UniTB-dense | 50,000 | 0.601× | 2.181× |
| UniTB-dense | 130,000 | 0.531× | 2.357× |
| UniTB-X1 | 20,000 | 1.843× | 1.842× |
| UniTB-X1 | 50,000 | 2.036× | 2.045× |
| UniTB-X1 | 130,000 | 2.173× | 2.175× |

`scalar` 是库的默认前向模式；本发布保持默认值。`indexed_sandwich_multi` 通过环境变量显式选择。该开关作用于 dense，X1 仍走相同的 activation fused-P0 与分组 GEMM；两组 X1 时间的差别包含独立测量波动。

<details>
<summary>完整计时、四分位、峰值显存与数值差</summary>

毫秒为中位数 [Q1, Q3]，四分位使用 inclusive 插值；显存为峰值分配 GiB。计时包含模型前向与均方 loss 反向，排除 JIT、输入复制、缓冲区恢复和进程检查。

| 模式 | 模型 | 有向边 | SO2CUDA | 前向 ms | 反向 ms | 合计 ms | GiB |
|---|---|---:|---|---:|---:|---:|---:|
| scalar | dense | 20,000 | 关 | 131.61 [131.31, 132.22] | 394.84 [394.31, 394.98] | 526.08 [525.61, 527.10] | 8.579 |
| scalar | dense | 20,000 | 开 | 506.78 [506.37, 507.37] | 227.52 [227.03, 228.34] | 734.27 [733.77, 735.25] | 8.732 |
| scalar | x1 | 20,000 | 关 | 161.27 [161.00, 161.61] | 431.28 [430.93, 431.76] | 592.73 [592.23, 594.18] | 12.423 |
| scalar | x1 | 20,000 | 开 | 121.01 [120.93, 121.24] | 200.75 [200.15, 201.60] | 321.70 [321.24, 322.56] | 13.305 |
| scalar | dense | 50,000 | 关 | 169.61 [169.21, 169.95] | 793.15 [792.62, 793.62] | 962.67 [961.86, 963.32] | 21.168 |
| scalar | dense | 50,000 | 开 | 1232.40 [1231.90, 1232.83] | 370.32 [369.69, 370.48] | 1603.07 [1602.15, 1603.52] | 21.550 |
| scalar | x1 | 50,000 | 关 | 220.78 [219.82, 221.61] | 858.89 [858.19, 860.51] | 1079.76 [1079.11, 1081.27] | 30.803 |
| scalar | x1 | 50,000 | 开 | 188.51 [188.06, 188.58] | 342.25 [340.73, 343.14] | 530.31 [529.74, 530.89] | 33.004 |
| scalar | dense | 130,000 | 关 | 314.58 [314.38, 314.79] | 1791.34 [1791.23, 1792.50] | 2105.96 [2105.60, 2107.43] | 54.739 |
| scalar | dense | 130,000 | 开 | 3198.45 [3196.99, 3199.77] | 767.73 [767.20, 768.00] | 3965.87 [3965.01, 3967.41] | 55.727 |
| scalar | x1 | 130,000 | 关 | 438.96 [438.83, 439.13] | 1946.04 [1945.41, 1946.50] | 2385.00 [2383.99, 2385.65] | 79.817 |
| scalar | x1 | 130,000 | 开 | 384.78 [384.42, 385.00] | 712.93 [712.66, 713.69] | 1097.55 [1096.98, 1098.57] | 85.533 |
| multi | dense | 20,000 | 关 | 130.69 [130.23, 130.89] | 394.74 [394.63, 395.11] | 525.56 [525.28, 525.84] | 8.579 |
| multi | dense | 20,000 | 开 | 106.87 [106.83, 107.01] | 178.53 [178.47, 179.21] | 285.72 [285.32, 286.06] | 9.011 |
| multi | x1 | 20,000 | 关 | 159.37 [159.24, 159.60] | 430.57 [430.08, 430.63] | 589.75 [589.48, 589.91] | 12.423 |
| multi | x1 | 20,000 | 开 | 120.11 [119.60, 120.35] | 199.81 [199.72, 200.57] | 320.13 [319.57, 320.92] | 13.305 |
| multi | dense | 50,000 | 关 | 168.35 [168.20, 168.59] | 793.23 [793.10, 793.43] | 961.71 [961.26, 962.15] | 21.168 |
| multi | dense | 50,000 | 开 | 154.28 [154.14, 154.46] | 286.65 [286.47, 286.73] | 440.95 [440.77, 441.20] | 22.249 |
| multi | x1 | 50,000 | 关 | 218.96 [218.84, 219.28] | 857.43 [857.24, 857.63] | 1076.66 [1076.20, 1077.24] | 30.803 |
| multi | x1 | 50,000 | 开 | 187.16 [187.08, 187.66] | 339.27 [339.07, 339.65] | 526.57 [526.15, 527.70] | 33.004 |
| multi | dense | 130,000 | 关 | 314.90 [314.56, 315.04] | 1790.03 [1789.82, 1790.19] | 2105.08 [2104.38, 2105.29] | 54.739 |
| multi | dense | 130,000 | 开 | 311.48 [310.30, 312.13] | 581.09 [579.91, 582.39] | 893.21 [890.06, 894.21] | 57.548 |
| multi | x1 | 130,000 | 关 | 438.02 [437.72, 438.78] | 1943.29 [1943.19, 1943.52] | 2381.54 [2380.98, 2382.19] | 79.817 |
| multi | x1 | 130,000 | 开 | 383.38 [382.62, 383.50] | 711.67 [711.47, 711.93] | 1094.95 [1094.09, 1095.38] | 85.533 |

| 模式 | 模型 | 有向边 | 训练输出最大绝对差 | 推理输出最大绝对差 | 梯度最大绝对差 | 超阈值梯度张量数 | 最大容差倍数 |
|---|---|---:|---:|---:|---:|---:|---:|
| scalar | dense | 20,000 | 3.64e-06 | 3e-06 | 1.34e-07 | 0 | 0.071 |
| scalar | x1 | 20,000 | 3.34e-06 | 3.58e-06 | 2.25e-06 | 0 | 0.852 |
| scalar | dense | 50,000 | 3.43e-06 | 3.34e-06 | 3.5e-07 | 0 | 0.199 |
| scalar | x1 | 50,000 | 4.29e-06 | 4.29e-06 | 6.21e-06 | 2 | 3.204 |
| scalar | dense | 130,000 | 3.25e-06 | 3.22e-06 | 2.66e-07 | 0 | 0.151 |
| scalar | x1 | 130,000 | 3.81e-06 | 3.34e-06 | 1.51e-05 | 9 | 6.391 |
| multi | dense | 20,000 | 2.98e-06 | 3.81e-06 | 1.23e-07 | 0 | 0.070 |
| multi | x1 | 20,000 | 4.29e-06 | 3.34e-06 | 2.26e-06 | 0 | 0.858 |
| multi | dense | 50,000 | 2.86e-06 | 3.1e-06 | 3.39e-07 | 0 | 0.192 |
| multi | x1 | 50,000 | 3.81e-06 | 4.77e-06 | 6.23e-06 | 2 | 3.201 |
| multi | dense | 130,000 | 3.81e-06 | 4.17e-06 | 2.63e-07 | 0 | 0.149 |
| multi | x1 | 130,000 | 3.81e-06 | 3.81e-06 | 1.51e-05 | 9 | 6.388 |

逐元素判据为 `max_abs_diff <= 1e-6 + 1e-5 * max_abs(reference)`。所有前向输出在容差内。大图 X1 的部分参数梯度超过该判据：原始严格失败状态保留；其差异来自 FP32 梯度累加顺序，未通过更改原回执或容差消除。

| 模式 | X1 有向边 | 最大绝对差张量的相对 L2 | 超阈值梯度张量中最大相对 L2 | 参考幅度 >1e-6 的梯度中最大相对 L2 | 全梯度张量原始相对 L2 最大值 |
|---|---:|---:|---:|---:|---:|
| scalar | 20,000 | 3.33e-06 | 0 | 1.16e-05 | 4.55e+17 |
| scalar | 50,000 | 1.16e-05 | 1.16e-05 | 3.12e-05 | 2 |
| scalar | 130,000 | 4.2e-05 | 5.5e-05 | 9.13e-05 | 9.38 |
| multi | 20,000 | 3.32e-06 | 0 | 1.15e-05 | 3.16 |
| multi | 50,000 | 1.16e-05 | 1.16e-05 | 3.12e-05 | 3.84 |
| multi | 130,000 | 4.2e-05 | 5.5e-05 | 9.14e-05 | 1 |

相对 L2 是逐张量值。接近零的电荷响应 bias 梯度会使原始相对比值很大（包括参考为 0 的情形）；它们的绝对差不超过 1.97e-10，均满足绝对容差。表中的参考幅度分组仅解释比值，不替代原始判据。13 万边 X1 最大绝对差来自末层 `node_update.tp.fc_m0.weight_shared`，相对 L2 约 4.20e-5；它并非所有梯度张量的相对 L2 最大值。

</details>

来源：固定 `h200_benchmark.py`、冻结版 `deeptb_speed_test.py` 与 `configs/dense.json` / `configs/x1.json`。配置使用兼容名 `lem_moe_v3_edge_h0`；与当前公开 `unitb` 简化配置的模型初态、参数键／形状和有效设置已逐项核对。当前公开示例没有单独在 H200 重新计时；本表来自固定 runner。合成模型结果不代表物理精度、真实数据训练吞吐或其他输入形状的性能。

被测 DeePTB：`137d6ce8b18b9142c145bc23586f5f31f4c81570`；SO2CUDA：`2f6e86bb5c72e8ceb44d8dfdc1e3e9e2b0b66e3e`，与本发布运行时源码相同。

- scalar：HP payload `c1a1b66b4baab8d8`；runner SHA256 `a4fb804cf5cdbe1d0adbd856cd92a25b87eebeb4f28a5adec6fb604acb76e03d`。
- indexed_sandwich_multi：HP payload `5b7c41e672da485e`；runner SHA256 `caf99eac2b8242f6cc4fc548e68a157057b934ea7bca545ef3929f579311a5f3`。

配置 SHA256：dense: `39f4b2e27dd40f6b85ed247e13dfc8eadb64f7d3097c46d0d5832bbec58ae72e`; x1: `7410c362aaa4e726fcb4488c6c84823d6807ed43b5690e23e6c0c7594ef65a26`。

<!-- FIN_H200_TABLE_END -->

## 构建与运行设置

| 环境变量 | 用途 |
| --- | --- |
| `CUDA_HOME` | CUDA Toolkit 目录 |
| `TORCH_EXTENSIONS_DIR` | PyTorch JIT 扩展缓存目录 |
| `SO2_CUDA_PACK_SCATTER_BUILD_DIR` | 张量打包与 scatter 扩展构建目录 |
| `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR` | 分组 GEMM 扩展构建目录 |
| `SO2_CUDA_BACKEND` | DeePTB 后端策略：默认 `auto`，`off` 强制纯 PyTorch |
| `SO2_CUDA_FORWARD_MODE` | dense 前向模式：`scalar` 或 `indexed_sandwich_multi` |
| `SO2_CUDA_FAST_TF32` | TF32 开关；上述示例固定为 `0` |
| `SO2_CUDA_PROFILE` | 算子分段计时开关；速度测试期间关闭 |
| `SO2_CUDA_MIN_EDGES` / `SO2_CUDA_MAX_EDGES` | DeePTB 扩展非 MoE dense 层的加速边数范围；`0` 不设限 |

包内 pack/scatter 和分组 GEMM 有各自的构建目录；需要把全部 JIT 产物放到指定位置时，同时设置这两个目录与 `TORCH_EXTENSIONS_DIR`。`CC` / `CXX` 可选择与 Toolkit 兼容的编译器，`MAX_JOBS` 控制并行编译数。DeePTB 的旧环境变量兼容入口见其 [SO2 后端文档](https://github.com/Franklalalala/DeePTB/blob/1006-stable/docs/so2_backend.md)。

通用张量积算子的最小示例见 [examples/minimal_so2_tp.py](examples/minimal_so2_tp.py)。
