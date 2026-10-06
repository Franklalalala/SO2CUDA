# SO2CUDA

SO2CUDA 为 SO(2) 张量积和专家线性层提供 CUDA 加速，包括分组 GEMM、旋转后的张量打包与输出 scatter，以及 DeePTB 的 dense / X1 执行接口。Python 包名为 `so2_cuda_ops`。

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

安装提供 `lem_moe_v3_edge_h0` 模型构建入口和 SO2CUDA 适配接口的 DeePTB 版本。例如，在本仓库与 DeePTB 的源码目录分别执行：

```bash
pip install -e .
pip install -e ../DeePTB
```

DeePTB 通过 `so2_cuda_ops.deeptb` 中的少数入口调用 SO2CUDA：`dense_pairs`、`activation_forward`、`grouped_gemm`、`grouped_gemm_multi` 和 `permute_rows`。接口只接受张量与数据描述对象，SO2CUDA 本身不导入 DeePTB。模型参数与检查点结构由 DeePTB 管理。

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

两个配置都保留三层网络、`lmax=6` 和生产隐藏通道：dense 使用单专家与均匀 32 通道；X1 使用四个路由专家、一个共享专家、top-k=2、共享低秩专家参数、V 形通道、先验 CG 路由与局域 QEq 响应头。示例采用两种元素的精简基组，包含 f 轨道以覆盖最高阶张量积。flow time 固定为 0；随机 H0 和合成 overlap 只用于性能测试，不代表物理先验或模型预测精度。

在本仓库根目录运行：

```bash
python examples/deeptb_speed_test.py --model dense --json dense_speed.json
python examples/deeptb_speed_test.py --model x1 --json x1_speed.json
```

默认每批约 20,000 条有向边：每个周期胞 125 个原子，每原子 80 个邻居。`--edges` 指定目标边数，脚本向上取整到完整周期胞并记录实际边数；`--side`、`--neighbors` 和 `--spacing` 可调节结构。默认 CUDA 分配器上限为 20 GiB，遇到显存不足时减小边数：

```bash
python examples/deeptb_speed_test.py --model both --edges 10000 --iterations 5 --json speed.json
```

脚本在同一模型、输入和参数上分别运行 PyTorch 参考路线与 CUDA 路线。参考路线显式使用 `staged` 张量积和 `split_loop` 专家线性层；CUDA 路线使用 fused-P0 和 cuBLAS 分组 GEMM。每轮恢复路由缓冲区，不做优化器更新。计时包含模型前向与 loss 反向，排除输入生成、复制、缓冲区恢复和首次 JIT 编译；默认先预热三轮，再记录十轮的中位数。

结果包括前向、反向与合计 ms/iter、加速比、峰值显存、实际成功调用的加速入口，以及推理输出、训练输出和参数梯度的最大绝对差与相对 L2 差。CUDA 模式会核验 fused 入口确实执行，未进入加速路线会报错。计算固定为严格 FP32，禁用 TF32；参考与 CUDA 路线的求和顺序可能造成 FP32 舍入差，JSON 会给出实测数值。

dense 前向模式自动选择：H200 使用 `scalar`，其他 GPU 使用 `indexed_sandwich_multi`。可显式指定模式：

```bash
python examples/deeptb_speed_test.py --model dense --forward-mode indexed_sandwich_multi
```

共用 GPU 的结果可能有计时噪声；用 `--shared-gpu` 将这一情况记入 JSON。CPU 或未安装 SO2CUDA 的 DeePTB 环境可以先做纯 PyTorch 小规模检查：

```bash
python examples/deeptb_speed_test.py --model both --device cpu --backend reference \
  --edges 64 --side 2 --neighbors 8 --warmup 0 --iterations 1 --json cpu_smoke.json
```

## 构建与运行设置

| 环境变量 | 用途 |
| --- | --- |
| `CUDA_HOME` | CUDA Toolkit 目录 |
| `TORCH_EXTENSIONS_DIR` | PyTorch JIT 扩展缓存目录 |
| `SO2_CUDA_PACK_SCATTER_BUILD_DIR` | 张量打包与 scatter 扩展构建目录 |
| `SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR` | 分组 GEMM 扩展构建目录 |
| `SO2_CUDA_FORWARD_MODE` | dense 前向模式：`scalar` 或 `indexed_sandwich_multi` |
| `SO2_CUDA_FAST_TF32` | TF32 开关；上述示例固定为 `0` |
| `SO2_CUDA_PROFILE` | 算子分段计时开关；速度测试期间关闭 |

通用张量积算子的最小示例见 [examples/minimal_so2_tp.py](examples/minimal_so2_tp.py)。
