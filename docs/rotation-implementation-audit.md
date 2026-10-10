# 旋转与 SO(2) 线性的源码核查

本页核查指定版本的源码调用链：将特征旋转到边坐标系，执行逐 m 的 SO(2) 线性，再旋转回原坐标系。这里的 **indexed sandwich** 指 SO2CUDA 用通道索引在 CUDA pack 中读取特征并计算所需 Wigner 分量，接分组 GEMM，再在 CUDA scatter 中合并复数线性结果、逆旋转与输出写入。GEMM 与 pack/scatter 是分别调用的阶段。它与把系数排列并入 Wigner 矩阵的优化有明确区别。

下表中的判断只覆盖所列版本、文件和公开调用链，不是文献新颖性结论，也不推断未读取的底层实现。

单算子测试以各实现自己的原生特征布局计时，规范输入、上游梯度与输出的布局转换均在计时区外；测量显存只保留当前实现自己的几何、参数和原生输入。

| 实现与版本 | 旋转 → SO(2) 线性 → 逆旋转 | 本页核查到的 indexed sandwich |
|---|---|---|
| EquiformerV3 `a7300c58`，原版 eager | 系数排列并入 Wigner；显式 `bmm`；逐 m `nn.Linear`；显式逆旋转 `bmm` | 所读原版调用链未使用 SO2CUDA 式 CUDA pack/scatter |
| cuEquivariance / cuEquivariance-torch `0.12.0` | `Rotation`；`escn_tp` 或 `escn_tp_compact` 描述符执行器；逆 `Rotation`。另测预计算 Wigner 的 PyTorch 旋转 | 公开组合是独立阶段；`indexed_linear` 自身有索引与 scatter 能力，不能据此说 cuEq 没有索引优化 |
| fairchem-core `2.10.0`，UMA 的 eSCNMD | 系数排列并入 Wigner；显式 `bmm`；逐 m Linear / MoLE；显式逆旋转 `bmm` | 所读 UMA 调用链未把通道 gather、Wigner pack 与逆旋转 scatter 组成 SO2CUDA 路线 |
| fairchem-core `2.10.0`，独立 eSEN / eSCN | 所核查的安装包中没有这些独立模块；不能用 UMA 的实现代替其证据 | 该版本安装包范围内不适用；下文另列可复核的历史源码 |
| e3nn `0.5.8` | 提供 Wigner / irreps 表示矩阵和一般 SO(3) 张量积；所读 API 不是完整 eSCN 型 SO(2) sandwich 层 | 在下文所列 Wigner 和 TensorProduct 代码路径中未见这一组合 |
| DeePTB 上游 main 快照 `1dcc7f6` | 逐 l `bmm`；布尔掩码提取逐 m 特征；逐 m Linear；写回；逐输出 irrep `einsum` 逆旋转 | 原版使用普通 PyTorch 提取和写回，没有所述 CUDA pack/scatter |
| SO2CUDA `484b7c7` 的 `dense_pairs` 默认路线 | 逐 m CUDA Wigner pack；多问题分组 GEMM；逐 m CUDA raw scatter；调用方按序相加 | 有；默认不是一个全算子 CUDA 核，也不是默认多 m 输出主序 scatter |

## EquiformerV3 原版

核查提交为 [`a7300c58df683dc99cb48027d5bfd4c887486c48`](https://github.com/atomicarchitects/equiformer_v3/tree/a7300c58df683dc99cb48027d5bfd4c887486c48)。`SO3Rotation` 的注释明确说明将 l→m 排列与旋转合并，也将 m→l 排列与逆旋转合并；构造映射和保存旋转矩阵见 [`so3.py:291–355`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/so3.py#L291-L355)。这项已有优化应保留在基线中。

`rotate` 和 `rotate_inv` 分别执行 `torch.bmm(self.wigner, inputs)` 与 `torch.bmm(self.wigner_inv, inputs)`，见 [`so3.py:358–368`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/so3.py#L358-L368)。构造阶段先分配所有 l 的完整系数 Wigner 矩阵并填入各 l 块，再转换到所需 m 排列，见 [`so3.py:401–424`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/so3.py#L401-L424)。矩阵作用在球谐系数轴上，通道作为另一维；它不是整个展开特征维度上的稠密矩阵。

局部线性层中，m=0 用 `fc_m0`，m>0 用 `SO2MLinear.fc`，随后组合实部和虚部；所有 m 的输出最后一次 `cat`。证据见 [`so2_ops.py:35–60`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/so2_ops.py#L35-L60) 和 [`so2_ops.py:111–156`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/so2_ops.py#L111-L156)。这条 eager 调用链有布局优化，但没有按通道索引执行 Wigner pack/scatter 的 CUDA 路线。

原版用单个 `num_in_channels` 和单个 `num_out_channels` 定义全部 l 的通道数，见 [`so2_ops.py:74–107`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/models/equiformer_v3/so2_ops.py#L74-L107)。因此，输入各 l 通道数统一、输出各 l 通道数也统一时，才能直接映射到这里核查的原版层；二者可以互不相同。V 形非均匀通道网格应标 N/A，不能补零后当作原版同形状基线。

本仓库的 [EquiformerV3 适配器](../examples/operator_eqv3.py) 校验原文件哈希，映射通道排列、复数权重符号和截断 m 的逆旋转缩放，将 m=0 偏置固定为零，然后直接调用原版层。eager 是本次主表的执行口径；原训练器另有可选 `torch.compile`，见 [`equiformer_v3_dens_trainer.py:379–380`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/experimental/trainers/equiformer_v3_dens_trainer.py#L379-L380)，不能把 eager 结果解释为其所有训练配置的最高性能。

## cuEquivariance 0.12.0

源码版本为 [`v0.12.0` / `70674ada`](https://github.com/NVIDIA/cuEquivariance/tree/70674ada7a00372e88ebe99969c700a6dcddd08f)。`Rotation` 构造 `yxy_rotation` 描述符和一个 `SegmentedPolynomial` 执行器，在前向中编码三个 Euler 角并调用该执行器；布局转置位于其输入和输出边界。证据见 [`rotation.py:68–90`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance_torch/cuequivariance_torch/operations/rotation.py#L68-L90)、[`rotation.py:122–158`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance_torch/cuequivariance_torch/operations/rotation.py#L122-L158) 和 [`rotations.py:51–85`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance/cuequivariance/group_theory/descriptors/rotations.py#L51-L85)。这里的角度旋转描述符本身可合并三个轴旋转；它不等同于将外部 SO(2) 权重和前后旋转合成一个 sandwich。

`escn_tp` 只描述局部 SO(2) 线性：对输入、输出 l 配对构造 cosine / sine 路径，归一化后返回以权重和局部特征为输入的描述符，见 [`escn.py:63–114`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance/cuequivariance/group_theory/experimental/escn.py#L63-L114)。它支持 SO3 和 O3；O3 分支有额外的宇称约束。本 benchmark 用 SO3 映射共同的 SO(2) 参数族。

`escn_tp_compact` 仅支持 SO3；按 m 建立片段，将该 m 下所有 l≥|m| 的通道放在一起，再构造 m=0 和正负 m 的线性路径。它没有接收 Wigner 或旋转角作为操作数。证据见 [`escn.py:117–181`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance/cuequivariance/group_theory/experimental/escn.py#L117-L181)。其参数映射必须使用实际归一化后的路径系数，不能假定系数都是 1。

本仓库 [cuEq 适配器](../examples/operator_baselines.py) 将旋转、描述符线性和逆旋转分别调用。compact 可在 `Rotation` 之后通过索引重排，或把 m 排列并入预计算 Wigner 再用 `bmm`；这两种公开拼法都纳入候选。描述符和 method 的可用性取决于形状：`uniform_1d` 检查操作数维数和片段一致性，见 [`segmented_polynomial_uniform_1d.py:119–153`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial_uniform_1d.py#L119-L153)。不支持的组合应保留原始错误。

cuEq 的 `indexed_linear` 确实支持输入索引、索引线性和输出 `scatter_add_`，见 [`segmented_polynomial_indexed_linear.py:130–151`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial_indexed_linear.py#L130-L151) 和 [`247–326`](https://github.com/NVIDIA/cuEquivariance/blob/70674ada7a00372e88ebe99969c700a6dcddd08f/cuequivariance_torch/cuequivariance_torch/primitives/segmented_polynomial_indexed_linear.py#L247-L326)。因此，本页只确认上述公开 sandwich 组合的阶段边界，不声称 cuEq 没有 gather/scatter，也不从 Python 包装推断其编译扩展内部的所有融合能力。

## fairchem-core 2.10.0：UMA 与版本边界

安装包元数据为 `fairchem-core==2.10.0`，对应公开标签 [`fairchem_core-2.10.0` / `ad6f4948`](https://github.com/facebookresearch/fairchem/tree/ad6f4948cc90accd9c2cd4758162f7b5979b5c8e)。所读 UMA Python 文件与该标签源码逐字节一致。

UMA 的 `_get_rotmat_and_wigner` 构造 Wigner 和其转置；截断 m 时对矩阵取子集，再用 `einsum` 把 `mappingReduced.to_m` 并入两侧矩阵，见 [`uma/escn_md.py:256–285`](https://github.com/facebookresearch/fairchem/blob/ad6f4948cc90accd9c2cd4758162f7b5979b5c8e/src/fairchem/core/models/uma/escn_md.py#L256-L285)。`Edgewise.forward_chunk` 先按节点索引取源、目标特征，再将二者拼接，执行显式 Wigner `bmm`、两层 SO(2) 卷积及中间激活、显式逆 Wigner `bmm`，最后 `index_add_` 到目标节点。证据见 [`uma/escn_md_block.py:209–238`](https://github.com/facebookresearch/fairchem/blob/ad6f4948cc90accd9c2cd4758162f7b5979b5c8e/src/fairchem/core/models/uma/escn_md_block.py#L209-L238)。节点 gather 与消息聚合 scatter 属于图消息传递，不是 SO2CUDA 的球谐通道 pack/scatter。

局部 `SO2_Convolution` 使用 `split` 取各 m，逐 m 调用 Linear，可先施加径向调制，最后 `cat`；复数线性结果用实部、虚部相加减。证据见 [`uma/nn/so2_layers.py:59–76`](https://github.com/facebookresearch/fairchem/blob/ad6f4948cc90accd9c2cd4758162f7b5979b5c8e/src/fairchem/core/models/uma/nn/so2_layers.py#L59-L76) 和 [`155–198`](https://github.com/facebookresearch/fairchem/blob/ad6f4948cc90accd9c2cd4758162f7b5979b5c8e/src/fairchem/core/models/uma/nn/so2_layers.py#L155-L198)。MoLE 替换层保留这个外层顺序；`MOLE.forward` 通过 `einsum` 混合专家权重，随后逐系统区间调用 `F.linear`，见 [`uma/nn/mole.py:169–206`](https://github.com/facebookresearch/fairchem/blob/ad6f4948cc90accd9c2cd4758162f7b5979b5c8e/src/fairchem/core/models/uma/nn/mole.py#L169-L206)。这些所读路径未使用 SO2CUDA 式 Wigner pack/scatter。

所核查的 **2.10.0 安装包**包含 `models/uma/` 中的 eSCNMD 和 MoE 类，但没有独立的 `models/esen/` 或 `models/escn/`；这两个名称不能都标成“2.10.0 已核查的独立卷积”。为给出可复核的实现证据，下面明确使用 EquiformerV3 提交 `a7300c58` 所带的历史 fairchem 源码，版本不与 2.10.0 混用。

| 补充历史源码 | 所读路径与证据 |
|---|---|
| eSEN，EquiformerV3 源码树 `a7300c58` | `Edgewise` 用被 mask 选择的 Wigner 矩阵 `bmm`，调用两层 SO(2) 卷积，再逆 `bmm`，见 [`esen_block.py:102–121`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/src/fairchem/core/models/esen/esen_block.py#L102-L121)。卷积在两端显式 `einsum` 变换 l/m 排列，中间逐 m Linear，见 [`nn/so2_layers.py:139–203`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/src/fairchem/core/models/esen/nn/so2_layers.py#L139-L203)。所读路径没有 CUDA indexed sandwich。 |
| eSCN，EquiformerV3 源码树 `a7300c58` | `SO3_Rotation.rotate/rotate_inv` 对 Wigner 取 mask 后 `bmm`，见 [`so3.py:381–391`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/src/fairchem/core/models/escn/so3.py#L381-L391)。`SO2Block` 两端显式 `_m_primary` / `_l_primary`，逐 m `SO2Conv`；该卷积用普通实部、虚部 Linear 加径向调制，见 [`escn.py:863–980`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/src/fairchem/core/models/escn/escn.py#L863-L980) 与 [`so3.py:216–221`](https://github.com/atomicarchitects/equiformer_v3/blob/a7300c58df683dc99cb48027d5bfd4c887486c48/src/fairchem/core/models/escn/so3.py#L216-L221)。所读路径没有 CUDA indexed sandwich。 |

## e3nn 0.5.8

核查范围是 PyTorch e3nn [`0.5.8`](https://github.com/e3nn/e3nn/tree/0.5.8) 中的 `o3.wigner_D`、`Irreps.D_from_angles/D_from_matrix` 和一般 `TensorProduct`。`wigner_D` 用三个生成元的矩阵指数相乘，见 [`_wigner.py:94–99`](https://github.com/e3nn/e3nn/blob/0.5.8/e3nn/o3/_wigner.py#L94-L99)；`Irreps.D_from_angles` 构造各 irrep 的矩阵并做 direct sum，见 [`_irreps.py:743–747`](https://github.com/e3nn/e3nn/blob/0.5.8/e3nn/o3/_irreps.py#L743-L747)。这些 API 生成表示矩阵，不自行执行完整 SO(2) sandwich。

`TensorProduct.forward` 将两个输入和权重传给生成的执行函数，见 [`_tensor_product.py:525–554`](https://github.com/e3nn/e3nn/blob/0.5.8/e3nn/o3/_tensor_product/_tensor_product.py#L525-L554)。代码生成器按指令生成带 Clebsch–Gordan 系数的收缩，见 [`_codegen.py:107–165`](https://github.com/e3nn/e3nn/blob/0.5.8/e3nn/o3/_tensor_product/_codegen.py#L107-L165)。所读路径中没有“边方向旋转 → 满逐 m 线性 → 逆旋转”的专用 gather/pack/scatter 调用链。一般 SO(3) 张量积的函数族和参数约束也不能自动等同于本 benchmark 的逐 m SO(2) 线性；e3nn 在这里用于独立 Wigner 与旋转核验，不作为未经参数等价映射的同算子性能列。本结论不覆盖 e3nn-jax 或未列出的实验模块。

## DeePTB 上游 main 与本仓库的纯 PyTorch 实现

核查上游 main 的固定快照 [`1dcc7f61480c373870cd5bad1d4000ac80757ff5`](https://github.com/deepmodeling/DeePTB/tree/1dcc7f61480c373870cd5bad1d4000ac80757ff5)。`SO2_Linear.forward` 计算所有 l 的 Wigner，将同 l 的 multiplicity 拼接做 `bmm`，再写回完整局部特征；见 [`tensor_product.py:194–231`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py#L194-L231)。随后通过布尔 mask 提取 m=0 和正负 m，逐 m 调 Linear，组合并写回，最后逐输出 irrep `einsum` 逆旋转，见 [`233–276`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py#L233-L276) 与 [`SO2_m_Linear:314–325`](https://github.com/deepmodeling/DeePTB/blob/1dcc7f61480c373870cd5bad1d4000ac80757ff5/dptb/nn/tensor_product.py#L314-L325)。它有普通 PyTorch 索引选择和写回，但没有 SO2CUDA CUDA Wigner pack/scatter。

本仓库 [单算子的纯 PyTorch 实现](../examples/operator_baselines.py) 保留上述计算顺序，使用计时区外预计算的 Wigner 与共同规范权重；[模型级纯 PyTorch 实现](../examples/naive_baseline.py) 还处理 UniTB 的径向调制与专家路由。它们属于“我们的纯 PyTorch 实现”，不是从 DeePTB 上游原文件直接调用的未修改基线，也不是 EquiformerV3 原版；它们不含这里定义的 indexed sandwich。

## SO2CUDA 的融合边界

以下按 SO2CUDA [`484b7c74bcf48910c6c95b744111d50617ea0682`](https://github.com/Franklalalala/SO2CUDA/tree/484b7c74bcf48910c6c95b744111d50617ea0682) 的 `dense_pairs` 公共入口核查。该入口返回 m>0 的贡献；m=0 由调用方先计算并在之后相加，见 [`deeptb.py:137–158`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/deeptb.py#L137-L158)。

1. **索引读取与 Wigner pack。** CUDA 根据 `in_base` 和 `in_l` 找到各通道的原始 irrep 块，计算所需 ±m 的 Wigner 收缩并写到成对特征缓冲区，见 [`so2_pack_scatter_kernel.cu:72–114`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/csrc/so2_pack_scatter_kernel.cu#L72-L114) 和 [`489–519`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/csrc/so2_pack_scatter_kernel.cu#L489-L519)。这一步将所需旋转分量的计算与 pack 合在一个 CUDA 核中，无须先生成完整 l 主序旋转特征再做 m 提取；pack 缓冲区本身仍然存在。
2. **分组 GEMM。** Python 将各 m 的成对特征展平成行，调用公开 `grouped_gemm_multi`；它进入独立的 cuBLAS 分组 GEMM，见 [`grouped_gemm.py:39–74`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/grouped_gemm.py#L39-L74) 和 [`cublas_grouped_gemm.cpp:66`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/csrc/cublas_grouped_gemm.cpp#L66)。GEMM 输出也仍然是中间张量；不是将旋转和 cuBLAS 放进同一个核。
3. **复数组合、逆旋转与 scatter。** raw scatter 直接读取 GEMM 的四个实值分量，组成两个 SO(2) 输出，用 Wigner 收缩并写到输出通道位置，见 [`so2_pack_scatter_kernel.cu:701–752`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/csrc/so2_pack_scatter_kernel.cu#L701-L752)。

**默认 `indexed_sandwich_multi` 是逐 m pack 和逐 m scatter，加一次多问题分组 GEMM。** `MULTI_PACK` 和 `MULTI_EPILOGUE` 的默认值都是 0；默认分支逐 m 调 `_PackPairFunction` 和 `_ScatterRawPairOutputFunction`，返回各 m 的贡献给调用方按序相加。证据见 [`tensor_product.py:2927–2930`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/tensor_product.py#L2927-L2930)、[`3047–3062`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/tensor_product.py#L3047-L3062) 和 [`3160–3177`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/tensor_product.py#L3160-L3177)。

多 m 合并 pack / epilogue 是显式开启的选项；输出主序 raw scatter 分支按输出 feature 遍历需要的 m 贡献后写入，见 [`tensor_product.py:3023–3042`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/tensor_product.py#L3023-L3042)、[`3098–3135`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/tensor_product.py#L3098-L3135) 和 [`so2_pack_scatter_kernel.cu:825–897`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/csrc/so2_pack_scatter_kernel.cu#L825-L897)。不能把这个可选分支当作默认路线，也不能把这里的 dense_pairs 结论直接套到所有 MoE 或 true-dense 入口。

[消融的显式旋转变体](../examples/operator_baselines.py) 将 m 排列并入预计算的完整系数 Wigner，在原生 `[E,(lmax+1)^2,C]` 布局上执行 `bmm`、一次 `split`、同一个公开 `grouped_gemm_multi`、复数结果组合与拼接，最后逆 `bmm`。它不调用 indexed sandwich pack/scatter，m=0 同样用普通 `F.linear`。因此，“EquiformerV3 原版 → 显式旋转 + 分组 GEMM → SO2CUDA 默认路线”记录的是两段实际实现变化；第二段包含旋转、复数结果组合、逆旋转和输出写入的调度与中间张量变化。时延和峰值显存需由该消融的实际数值给出，源码核查本身不证明收益大小。

## 非 MoE dense 与 grouped 候选

同一版本的公开 `true_dense_pairs` 接受非路由的二维权重。无径向调制时，它一次调用多 m Wigner pack，逐 m 使用 `F.linear`，最后一次输出主序 raw scatter；m=0 仍由调用方计算。证据见 [`_dense.py:40–124`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/_dense.py#L40-L124)。它适用于非 MoE dense 层，不使用上述默认路线的分组 GEMM；因此不把它替代到“显式旋转＋分组 GEMM → 默认 indexed sandwich”的消融链中。

`dense_pairs` 的 `forward_mode="indexed_sandwich_multi_grouped"` 属于公开候选。该版本入口用此参数决定是否进入多 m 路线，但内部的 grouped pack/epilogue 选择重新读取环境变量，见前述 `deeptb.py:149–158` 与 `tensor_product.py:2927–2930`。基准在计时区外同步设置公开 `SO2_CUDA_FORWARD_MODE`，使用结束后恢复环境；环境别名定义见 [`config.py:63–80`](https://github.com/Franklalalala/SO2CUDA/blob/484b7c74bcf48910c6c95b744111d50617ea0682/src/so2_cuda_ops/config.py#L63-L80)。等价性工具同时统计实际 pack/scatter 调用，确认三种候选各自执行了预期路线。性能主表取正确候选中前向＋反向中位数最小者，并保留全部计时。
