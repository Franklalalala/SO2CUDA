# 更新记录

## 未发布

- 增加单个 SO(2) 卷积的 EquiformerV3 原版基线、我们自己的纯 PyTorch 实现和 cuEquivariance 对照，核对输出、输入梯度、参数梯度及旋转等变性。
- 增加显式 Wigner 旋转加分组 GEMM 的消融，比较 indexed sandwich 的时间和峰值显存，并给出各实现的源码路径说明。
- 单算子以各实现的原生特征布局计时，布局转换放在计时区外；峰值显存只包含当前实现的数据与几何量。
- 增加 UniTB 的 cuEquivariance 执行示例，层间共享每次前向构建的 Wigner，保留原模型参数、径向调制与专家混合顺序。
- 公开 SO2CUDA pair 接口与 cuEquivariance 整模型组合按实际测试规模选择，保留所有候选计时和错误。
- 性能表由 JSON 生成，分别呈现单算子与模型测试，记录严格 FP32 下的时间和峰值显存。

## 0.2.0（2026-10-07）

- `so2_cuda_ops.deeptb` 提供张量与布局描述符接口，覆盖 dense SO2、激活空间专家路由和分组线性层。
- `dense_pairs` 支持 UniTB-dense；`activation_forward` 支持 UniTB 的 PDQ-MoE；`true_dense_pairs` 支持扩展的非 MoE dense SO2 层。DeePTB `1006-stable` 通过可选后端统一调用这些接口。
- SO2CUDA 统一维护 pack、分组 GEMM、scatter 和行置换的自动微分实现；分组 GEMM 支持 JVP。
- CUDA 源仅保存在 `src/so2_cuda_ops/csrc/`。
- 提供 UniTB 与 UniTB-dense 的合成周期结构加速示例，以同参数的纯 PyTorch 朴素实现（上游 SO2 张量积 + UMA 式专家层）为基线，报告严格 FP32 下的前向／反向时间、峰值显存与数值差。
