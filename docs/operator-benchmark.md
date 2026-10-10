# 单算子对照约定

测试边上的 eSCN 式卷积：\(y_e=D(R_e)^\mathsf{T}\mathcal L_W(D(R_e)x_e)\)。权重在边之间共享，不含径向调制。代码存储行向量，几何对象存从主轴到边方向的旋转；输入右乘它的 Wigner 矩阵，输出右乘转置。几何在计时外按各实现需要的格式准备，不求导。

## 参数映射

数值核验的规范特征使用 e3nn `mul_ir` 布局，依次为 irrep 块、通道、m。各实现以各自原生特征布局计时，输入与上游梯度的布局转换放在计时区外，输出保留原生布局；核验时再转换回规范布局。每个 m 的通道按 irreps 中的块顺序排列，保留 `|m| <= mmax`。规范参数为 `W[0]=W0`、`W[m]=cat([A_m,B_m],dim=0)`，正 m pair 顺序为 `(-m,+m)`：

\[
y_0=W_0x_0,\qquad y_-=A_mx_- - B_mx_+,\qquad y_+=A_mx_+ + B_mx_-.
\]

单算子各实现的 m=0 bias 固定为零。规范参数映射成各实现自己的可训练参数，计时包含原生参数的梯度；映射及梯度拉回用于设置和核验，放在计时外。

- SO2CUDA 使用公开 `prepare_layout`、`prepare_wigner` 与 pair 接口，均设 `radial_parts=None` 与 `include_m0=True`：m=0 与其余 m 在同一次公开调用中计算，返回整层输出。首页的 SO2CUDA 列是通用入口 `true_dense_pairs`（非路由的 `LinearWeights`，适用于非 MoE dense 层）。另测 `dense_pairs` 的默认与 `forward_mode="indexed_sandwich_multi_grouped"` 两种调用（共享权重对应一个路由组），以及单专家的 `activation_forward`（一个专家、top-1、门控为 1，`--impl activation`）；它们执行同一套 kernel，逐配置计时见下文“各入口计时”。每个入口计时前先核验等价性，JSON 保留全部入口的计时。
- 我们自己的纯 PyTorch 实现按 [DeePTB 上游 `SO2_Linear`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py) 写法实现，与 [naive_baseline.py](../examples/naive_baseline.py) 来源一致：输入按 l 合并后 `bmm`，按 m 掩码选择特征，`F.linear` 后用 `narrow` 组合实虚部，再按输出 irrep 旋转。
- EquiformerV3 直接导入[固定提交的原代码](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3)，执行 `SO3Rotation.set_wigner/rotate/rotate_inv` 和 `SO2Linear`，不改源码。计时区仅执行 `rotate_inv(linear(rotate(x)))`，输入、输出均为原生 `[E,(lmax+1)^2,C]`。原实现 pair 顺序为 `(+m,-m)`，原生权重为 `[A_m;-B_m]`，通道按 l 重排。初始化的 `1/sqrt(2)` 不属于前向运算。截断 m 时，原 `rotate_inv` 对 `l>mmax` 乘 `s_l=sqrt((2*l+1)/(2*mmax+1))`，因此原生权重的对应输出行除以 `s_l`。梯度拉回使用相同的变号、置换和 `1/s_l` 因子。
- cuEquivariance 使用公开 `escn_tp(cue.SO3 irreps)`、`escn_tp_compact(cue.SO3 irreps)` 与 `SegmentedPolynomial`。前者局部特征为 `ir_mul`，后者按 m 分段、段内 l 连续。PyTorch 旋转将描述符排列并入预计算 Wigner 矩阵：均匀通道使用原生 `[E,系数,C]` 和一次 `bmm`；一般 irreps 采用按 multiplicity 分组的原生布局，通过 `split` 分块。公开 `Rotation` 候选直接读写 `ir_mul`，不在计时区调用 `TransposeIrrepsLayout`；compact 所需的局部 m 排列在两侧旋转之间执行。JSON 记录实际布局。两种描述符表达同一个 SO(2) 线性函数族。SO3 保留每个正 m 的余弦及正弦自由度；O3 的宇称约束无法表达这里全部自由权重。按 `normalize_paths_for_operand(2)` 取得实际路径系数 `c_p`，规范矩阵元在该路径的符号为 `sigma_p`，原生参数为 `w_p[u,v]=sigma_p*W_m[v,u]/c_p`。映射读取公开描述符的路径和分段，核对共享同一权重的所有路径一致。梯度乘回 `sigma_p/c_p` 并还原到规范矩阵。

朴素 PyTorch SO(2) 基线使用 EquiformerV3 原版。它要求输入、输出分别覆盖完整的 l 范围，且各 l 通道数统一；不符合的单算子配置记 N/A。作者的直接训练配置有 `use_compile=True`，训练入口使用 `torch.compile(..., dynamic=True)`。首页同时列出 eager 与 `torch.compile`（`--include-compile`）两列：编译耗时单独记录，计时前比较编译与 eager 的输出及全部梯度。

## 与 EquiformerV3 原版的关系

EquiformerV3 原版本身就是显式旋转：把系数排列并入稠密 Wigner 矩阵后用 `bmm` 旋转，再做逐 m 线性和逆旋转。SO2CUDA 与它数学相同，差别在 indexed sandwich 的实现：CUDA kernel 以（边，通道）为单位，把输入的各 l 分量旋入每个 m 一块的缓冲（m>0 块每行为 `[x_{-m} | x_{+m}]`），每块做一次 GEMM，2×2 复结构写成实数块权重 `[[A,-B],[B,A]]`，再从各块旋回并写出特征；反向由同样的 kernel 与 GEMM 完成，不显式存储完整的 Wigner 矩阵或旋转后的整段特征。

## 计时与数值核验

前向在无梯度模式执行；前向加反向对输入及全部权重求导。正式计时至少预热 5 次、记录 20 次，报告 CUDA 同步计时的中位数、Q1/Q3 与峰值显存。计时前后核验 GPU 的计算及图形进程，要求独占。严格 FP32、关闭 TF32，cuEquivariance 使用 `math_dtype=torch.float32`。

测量每个实现时，显存只保留它自己的几何、布局索引、参数、原生输入与上游梯度。规范数据保存在 CPU，布局转换的临时副本在测量前释放，其他实现的参数和几何不同时驻留。峰值包含上述常驻张量以及输出、输入和参数梯度、反向保存的激活与工作空间；JSON 的 `memory_scope` 说明范围。每个实现开始前清理分配器缓存，reserved 峰值仍包含该次执行的分配器复用。

不加 `--cueq-choice` 时，工具对每个配置尝试两种描述符与 `naive`、`uniform_1d`、`fused_tp`、`indexed_linear` 的组合，并比较公开 `Rotation` 与 PyTorch Wigner `bmm`；候选通过前向、输入梯度、规范权重梯度与旋转等变检查后再测完整算子，按前向加反向中位数选最快组合。首页的 cuEquivariance 列用 `--cueq-choice escn_tp_compact,naive,pytorch` 只测这一组合：在全部 20 个配置的完整扫描中它都是最快的正确组合。该组合计时前同样先过数值核对。OOM 按原形状记录。

数值核验包括输出、输入梯度、映射回规范参数的逐 m 权重梯度、实现间两两比较及 float64 参考；另对输入和边向量作整体随机旋转，核对旋转等变性。判据允许合理的 FP32 累加顺序差异。

## 各入口计时

SO2CUDA 各公开入口在首页同一会话中的前向＋反向中位数（ms），配置与首页相同：

<!-- SO2CUDA_ENTRY_TIMINGS_BEGIN -->
| 配置 | 有向边 | `true_dense_pairs` ms | `dense_pairs` ms | `dense_pairs`（grouped） ms | `activation_forward`（单专家） ms |
|---|---:|---:|---:|---:|---:|
| ℓ=2, C=32, m≤2 | 50,000 | 0.93 | 1.04 | 1.02 | 1.02 |
| ℓ=2, C=64, m≤2 | 50,000 | 1.64 | 1.70 | 1.68 | 1.68 |
| ℓ=2, C=128, m≤2 | 50,000 | 4.18 | 4.24 | 4.23 | 4.24 |
| ℓ=4, C=32, m≤4 | 50,000 | 2.21 | 2.28 | 2.27 | 2.28 |
| ℓ=4, C=64, m≤4 | 50,000 | 5.60 | 5.68 | 5.65 | 5.67 |
| ℓ=4, C=128, m≤4 | 50,000 | 17.39 | 17.46 | 17.43 | 17.45 |
| ℓ=6, C=32, m≤6 | 50,000 | 5.97 | 6.07 | 6.05 | 6.06 |
| ℓ=6, C=64, m≤6 | 50,000 | 15.32 | 15.42 | 15.40 | 15.40 |
| ℓ=6, C=128, m≤6 | 50,000 | 47.64 | 47.71 | 47.50 | 47.70 |
| ℓ=6, C=128, m≤2 | 50,000 | 34.35 | 34.39 | 34.29 | 34.42 |
| ℓ=6, C=32, m≤6 | 20,000 | 2.67 | 2.76 | 2.75 | 2.75 |
| ℓ=6, C=32, m≤6 | 130,000 | 14.86 | 14.94 | 14.94 | 14.94 |
| ℓ=6, C=128, m≤6 | 20,000 | 19.76 | 19.88 | 19.74 | 19.89 |
| ℓ=6, C=128, m≤6 | 130,000 | 127.06 | 126.67 | 126.64 | 128.02 |
| V 形通道 | 20,000 | 3.70 | 3.79 | 3.77 | 3.78 |
| V 形通道 | 50,000 | 8.73 | 8.82 | 8.80 | 8.81 |
| V 形通道 | 130,000 | 22.14 | 22.27 | 22.25 | 22.25 |
| 输入／输出 irreps 不同 | 20,000 | 1.10 | 1.25 | 1.23 | 1.24 |
| 输入／输出 irreps 不同 | 50,000 | 1.38 | 1.44 | 1.42 | 1.43 |
| 输入／输出 irreps 不同 | 130,000 | 3.13 | 3.20 | 3.18 | 3.19 |

其他入口相对 `true_dense_pairs` 的前向＋反向时间比为 0.997–1.134。
<!-- SO2CUDA_ENTRY_TIMINGS_END -->

## 模型对照

[cueq_baseline.py](../examples/cueq_baseline.py) 通过标准子模块替换接入 UniTB，保留原参数对象和检查点键。径向调制位置、路由、激活、电荷头及其余普通专家线性层与我们的纯 PyTorch 实现一致。PDQ-MoE 按被选边计算各专家 SO(2) 输出，再按原门控权重累加，共享专家单独计算；混合保留在激活之前。模型参数映射在每次前向执行，其耗时计入模型测试。

cuEquivariance 路线每次完整前向只构造一次逐 l 的 compact Wigner，并通过 `wigner_D_all` 传给各 SO(2) 层。旋转使用按 l 合并的 `split`／`bmm`，局部特征接入所选描述符的原生排列；不逐层重建 Wigner，不逐 irrep 切片写回。模型测试核对 Wigner 构造数、缓存命中数和每层实际调用数。

执行替换应安装在模型完成 FP32 设备放置之后；移动设备前先恢复原模块。`--backend cueq-check` 核对默认路线与 cuEquivariance 的训练输出、推理输出和所有参数梯度，并确认 cuEquivariance 路线没有调用 SO2CUDA 入口。

模型的 cuEquivariance 列在全部六个 SO(2) 层统一使用 `escn_tp_compact` 与 `naive` method。每列先做一次不计时的前向＋反向，再按 3 次预热、10 次计时测量；每次计时迭代前后核验独占。SO2CUDA 列核对 DeePTB 公开入口的实际调用次数，纯 PyTorch 与 cuEquivariance 列核对没有调用 SO2CUDA。

完整 UniTB-dense 与 UniTB 的 EquiformerV3 路线标 N/A。按 l 合并各层通道后，UniTB-dense 最后一层输出为 `[151,37,41,29,13,5,1]`，节点末层输入为 `[57,69,73,61,45,37,33]`；这些非零 l 的通道数不统一。原版的 `extra_m0_out_channels` 可以表示前面层的额外标量 gate，但不能表示末层的一般通道分布。UniTB 的 V 形通道也不统一。核查不补零、不裁剪通道、不修改第三方源码。

首页与公开证据由 [update_readme_benchmarks.py](../examples/update_readme_benchmarks.py) 从一次测量会话的 JSON 生成。生成器要求矩阵中每个配置的全部实现都已测量、cuEquivariance 组合固定且通过数值核对、每个计时都有前后独占证据，并要求模型适用性与整模型数值证据；原始 JSON 保留报错原文，公开 JSON 保留去掉内部路径后的报错文本。
