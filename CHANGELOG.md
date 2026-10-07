# 更新记录

## 0.2.0（2026-10-07）

- dense fused-P0 前向默认改为 `indexed_sandwich_multi`（独占 H200 上比 `scalar` 快 2.6–4.4 倍）；需要与旧运行逐位一致时设 `DPTB_SO2_MOE_FUSED_P0_FORWARD_MODE=scalar`。
- `so2_cuda_ops.deeptb` 提供张量与布局描述符接口，覆盖 dense SO2、激活空间专家路由和分组线性层。
- `true_dense_pairs` 支持扩展的非 MoE dense SO2 层；DeePTB `1006-stable` 通过可选后端统一调用这些接口。
- SO2CUDA 统一维护 pack、分组 GEMM、scatter 和行置换的自动微分实现；分组 GEMM 支持 JVP。
- 删除重复顶层 CUDA 源、未用于生产配置的 scheduler、CUTLASS smoke 与 flat-backward 实验。
- CUDA 源仅保存在包内；提供 UniTB-dense 与 UniTB-X1（PDQ-MoE）的安装自检和合成周期结构加速示例，报告严格 FP32 的前向 / 反向中位数、四分位、峰值显存与数值差。
