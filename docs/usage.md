# 使用接口

DeePTB 集成入口为 `so2_cuda_ops.deeptb`。模型构造、径向网络、专家参数化与检查点由 DeePTB 管理；此入口只接收普通张量和小型 dataclass。

| 接口 | 用途 |
|---|---|
| `prepare_layout`、`PairLayout` | 从 `(l, multiplicity, first_feature)` 构造各 m 的整数映射；同一层、设备可复用 |
| `prepare_wigner`、`WignerData` | 接收 dense Wigner 张量或逐 l 的 compact blocks；几何不求导 |
| `dense_pairs`、`DenseRouting` | 返回按 m 排序的 m>0 输出贡献；调用方先形成 m=0，再依次相加 |
| `activation_forward`、`ActivationRouting`、`LinearWeights` | 激活空间 top-k 路由的 pack、分组线性层与输出 scatter |
| `grouped_gemm`、`grouped_gemm_multi` | 可微的分段矩阵乘法，分别用于单组问题与多个 m 块 |
| `permute_rows` | 双射行置换，反向用逆置换读取 |

生产 CUDA 路线采用 FP32。`x` 为 `[N, Din]`；分组线性层权重为 `[G, Dout, Din]`；`ptr` 是长度 `G+1`、覆盖输入行的 int64 前缀指针。CUDA 浮点张量须位于同一设备。接口接受普通非连续输入并在所需处转为连续布局。

`activation_forward` 的 `indices/values` 为 `[N,K]`，`slots` 每项为稳定排序的 `(order, inverse, ptr_cpu, sorted_expert_ids)`；`LinearWeights.weight` 为 `[experts,Dout,Din]`，非路由插值块为 `[Dout,Din]`。调用方已按模型定义折入共享专家时设置 `coefficients_sum_to_one=True`；否则传入未折入的共享权重。

`dense_pairs` 的 `weights` 与 `radial_parts` 都按 m 编号，包含 m=0 占位。m>0 权重的输出维为 `2*Cout_m`；径向块为 `[N,Cin_m]`（front）或 `[N,Cout_m]`（back）。`DenseRouting.ptr` 描述展平后的 `[N,2]` pair 行，必要时携带对应双射置换。

CPU、非 FP32、autocast、`torch.func` 或 Wigner 求导时，SO2 接口返回 `None`，由 DeePTB 使用纯 PyTorch 参考路线。分组 GEMM 直接调用要求 CUDA FP32，单问题接口额外支持 JVP。内核执行异常直接向上传播。

安装用 `pip install -e .`。`is_available()` 只报告 PyTorch 是否检测到 CUDA；第一次真正调用会 JIT 编译，也会验证本机工具链可用。无 CUDA 时可以正常导入包。

环境变量：`SO2_CUDA_PACK_SCATTER_BUILD_DIR`、`SO2_CUDA_CUBLAS_GROUPED_BUILD_DIR` 设置私有构建目录；`SO2_CUDA_FAST_TF32=0` 保持 FP32。DeePTB 适配层支持 `SO2_CUDA_BACKEND=off` 回退。公开示例见仓库首页。
