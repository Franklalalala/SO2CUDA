# 更新记录

## 0.2.0（2026-10-07）

- `so2_cuda_ops.deeptb` 提供张量与布局描述符接口，覆盖 dense SO2、激活空间专家路由和分组线性层。
- `dense_pairs` 支持 UniTB-dense；`activation_forward` 支持 UniTB 的 PDQ-MoE；`true_dense_pairs` 支持扩展的非 MoE dense SO2 层。DeePTB `1006-stable` 通过可选后端统一调用这些接口。
- SO2CUDA 统一维护 pack、分组 GEMM、scatter 和行置换的自动微分实现；分组 GEMM 支持 JVP。
- CUDA 源仅保存在 `src/so2_cuda_ops/csrc/`。
- 提供 UniTB 与 UniTB-dense 的合成周期结构加速示例，以同参数的纯 PyTorch 朴素实现（上游 SO2 张量积 + UMA 式专家层）为基线，报告严格 FP32 下的前向／反向时间、峰值显存与数值差。
