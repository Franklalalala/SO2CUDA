# 实现归属

SO2CUDA 维护 SO2 的 pack／scatter 原生核、分组 cuBLAS GEMM、自动微分、激活空间 top-k 编排、布局描述符与 JIT 工具链发现。唯一 CUDA 源目录为 `src/so2_cuda_ops/csrc/`。

DeePTB 保留模型类、专家参数、径向网络、路由概率计算、Wigner 几何构造、纯 PyTorch 参考数学、训练配置和检查点加载。H0 先验、晶体对称投影、e3nn 参考模块和激活重计算属于模型侧。

核心算子的既有 Python 入口继续用于独立算子调用；新集成使用 `so2_cuda_ops.deeptb`。未用于生产的 scheduler、CUTLASS smoke 和 flat-backward 实验不在此版本中。
