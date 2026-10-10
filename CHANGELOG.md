# 更新记录

## 0.3.1（2026-10-11）

- 旋入与收集增加按边分块的 kernel：每条边一个线程块，输入行、Wigner 块和输出行经共享内存暂存，读写连续。混合阶数的 warp（如 `24x1o+16x2e+...`）也合并读写。
- top-k 路由的 `activation_forward` 中，各槽共用一次旋入和一次收集：旋入一次写出 K 份按槽排序的副本，收集按门控加权求和后只旋回一次，反向同样各一次。各行的专家组号直接取自路由结果，不再每层把 CPU 端的段指针展开到 GPU。
- 前置径向权重在旋入 kernel 中施加，后置径向权重在收集 kernel 中施加。带径向权重的层前向只保存输入（后置径向层另存径向缩放前的求和），反向重算旋入。以 [unitb_slem.json](examples/configs/unitb_slem.json) 在 2 万条边上训练一步，峰值显存 16.15 → 12.70 GiB。
- 新增 UniTB-SLEM 合成配置 [examples/configs/unitb_slem.json](examples/configs/unitb_slem.json)。
- 接口与数值约定不变。README 中的性能表仍是 0.3.0 的测量结果。

## 0.3.0（2026-10-10）

- 新的块布局 sandwich 内核：每个 m 一块缓冲，(−m, +m) 成对存放，m>0 块的 2×2 复结构写成一个实数块权重，每块做一次 GEMM。旋入与收集各由一个按（边，通道）并行的 kernel 完成，反向复用同一套 kernel，不使用原子加。
- `true_dense_pairs` 与 `dense_pairs` 增加仅关键字参数 `include_m0`：设为 `True` 时，m=0 项（含偏置）在同一次调用中完成；默认 `False`，行为与 0.2.0 相同。
- 单专家与 top-k 路由的 `activation_forward`、按组的 `dense_pairs` 使用同一套块布局内核，径向权重与逐边门控在内核中处理。
- FP32 下单组 GEMM 逐问题调用 cuBLAS；分组 GEMM 的反向只计算需要的梯度。
- 增加单个 SO(2) 卷积的对照：EquiformerV3 原版（eager 与 `torch.compile`）、按 DeePTB 上游写法的纯 PyTorch 实现、cuEquivariance 0.12.0；逐项核对输出、输入梯度、参数梯度与旋转等变性。
- 性能表由同一块 H200、同一次会话的 JSON 生成，分为单算子与 UniTB 模型两节，记录严格 FP32 下的时间与峰值显存。

## 0.2.0（2026-10-07）

- `so2_cuda_ops.deeptb` 提供张量与布局描述符接口，覆盖 dense SO2、激活空间专家路由和分组线性层。
- `dense_pairs` 支持 UniTB-dense；`activation_forward` 支持 UniTB 的 PDQ-MoE；`true_dense_pairs` 支持扩展的非 MoE dense SO2 层。DeePTB `1006-stable` 通过可选后端统一调用这些接口。
- SO2CUDA 统一维护 pack、分组 GEMM、scatter 和行置换的自动微分实现；分组 GEMM 支持 JVP。
- CUDA 源仅保存在 `src/so2_cuda_ops/csrc/`。
- 提供 UniTB 与 UniTB-dense 的合成周期结构加速示例，以同参数的纯 PyTorch 朴素实现（上游 SO2 张量积 + UMA 式专家层）为基线，报告严格 FP32 下的前向／反向时间、峰值显存与数值差。
