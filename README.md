# SO2CUDA

SO2CUDA 为 SO(2) 张量积和专家线性层提供 CUDA 加速，包括分组 GEMM、旋转后的张量打包与输出 scatter，以及 DeePTB 的 UniTB / UniTB-dense 执行接口。UniTB 使用 PDQ-MoE 专家层。Python 包名为 `so2_cuda_ops`，当前版本为 **0.2.0**。

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

## UniTB 与 UniTB-dense 加速测试

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

**基线**在同一块 GPU 上使用同一模型、参数和输入，关闭 SO2CUDA（`SO2_CUDA_BACKEND=off`），由 [naive_baseline.py](examples/naive_baseline.py) 换成直接的 PyTorch 写法：SO2 张量积照 [DeePTB 上游主分支的 `SO2_Linear`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py)，专家层照 [UMA 的 MoLE](https://github.com/facebookresearch/fairchem/blob/3801dac0cc0458a2f8121259a2ce8b23d4dcc5a1/src/fairchem/core/models/uma/nn/mole.py) 线性公式。UniTB 的每个路由专家只对分给它的边做一次普通 `F.linear`，再按路由系数加权，共享专家单独计算；UniTB-dense 按路由组混合权重，每组做一次普通 `F.linear`。路由器、电荷头和其余层两条路线相同。

**加速路线**使用安装后的默认设置，脚本自动恢复原模块并选择 `SO2_CUDA_BACKEND=auto`。每轮恢复路由缓冲区，不做优化器更新。计时包含模型前向与 loss 反向，排除输入生成、复制、缓冲区恢复和首次 JIT 编译；默认预热 3 次、计时 10 次，报告中位数与 Q1 / Q3。计算固定为严格 FP32，禁用 TF32。

JSON 同时记录峰值显存、实际成功调用的加速入口、推理与训练输出和全部参数梯度的最大绝对差及逐张量相对 L2。基线调用 SO2CUDA 或加速路线未进入对应入口时会报错。共用 GPU 的结果可用 `--shared-gpu` 标记。

### H200 实测

NVIDIA H200（计时前后核验独占），严格 FP32、关闭 TF32，随机初始化模型与合成周期结构，同参数、同输入，每路预热 3 次、计时 10 次；下表为前向＋反向合计中位数，加速倍数为基线时间除以 SO2CUDA 时间。

| 模型 | 有向边 | 基线 ms | SO2CUDA ms | 加速倍数 |
|---|---:|---:|---:|---:|
| UniTB-dense | 20,000 | 553.74 | 282.84 | 1.96× |
| UniTB-dense | 50,000 | 1016.89 | 437.13 | 2.33× |
| UniTB-dense | 130,000 | 2230.38 | 887.79 | 2.51× |
| UniTB | 20,000 | 661.75 | 318.77 | 2.08× |
| UniTB | 50,000 | 1194.16 | 525.67 | 2.27× |
| UniTB | 130,000 | 2608.04 | 1095.05 | 2.38× |

训练／推理输出与全部参数梯度的最大绝对差约 1.5×10⁻⁵（FP32 舍入量级）。

合成模型结果不代表物理精度、真实数据训练吞吐或其他输入形状的性能。

## 构建与运行设置

- `CUDA_HOME`：CUDA Toolkit 目录。
- `TORCH_EXTENSIONS_DIR`：PyTorch JIT 扩展缓存目录。
- `SO2_CUDA_PACK_SCATTER_BUILD_DIR`：张量打包与 scatter 扩展构建目录。
- `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR`：分组 GEMM 扩展构建目录。
- `SO2_CUDA_BACKEND`：DeePTB 后端策略，默认 `auto`，`off` 强制纯 PyTorch。
- `SO2_CUDA_FAST_TF32`：TF32 开关；上述示例固定为 `0`。

需要把全部 JIT 产物放到指定位置时，同时设置两个算子构建目录与 `TORCH_EXTENSIONS_DIR`。`CC` / `CXX` 可选择与 Toolkit 兼容的编译器，`MAX_JOBS` 控制并行编译数。更多接口约定见 [docs/usage.md](docs/usage.md)。

通用张量积算子的最小示例见 [examples/minimal_so2_tp.py](examples/minimal_so2_tp.py)。
