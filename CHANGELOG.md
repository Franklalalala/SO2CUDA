# 更新记录

## 0.2.0

- `so2_cuda_ops.deeptb` 提供张量与布局描述符接口，覆盖 dense SO2、激活空间专家路由和分组线性层。
- SO2CUDA 统一维护 pack、分组 GEMM、scatter 和行置换的自动微分实现；分组 GEMM 支持 JVP。
- 删除重复顶层 CUDA 源、未用于生产配置的 scheduler、CUTLASS smoke 与 flat-backward 实验。
- CUDA 源仅保存在包内；提供 dense 与 X1 的安装自检和合成周期结构加速示例。
