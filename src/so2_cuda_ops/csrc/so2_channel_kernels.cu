// Channel-major SO(2) pack / scatter kernels.
//
// One thread owns one (edge, irrep channel) of degree l. It keeps the 2l+1
// rotated coefficients of that channel in registers, so every coefficient of
// the feature row is read or written exactly once and every Wigner entry of
// the edge's l block is shared by all channels of that degree (warp broadcast).
// The m blocks a channel belongs to are listed in a small per-channel table
// (absolute column of the channel in each m block, -1 when absent).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kChannelThreads = 128;

struct WignerRef {
  const float* __restrict__ data;
  const int64_t* __restrict__ offsets;          // dense mode: first coefficient of each l
  const int64_t* __restrict__ compact_offsets;  // compact mode: first entry of each l block
  int64_t dense_stride;
  int64_t compact_stride;
  int mode;  // 0 identity, 1 dense [N, D, D], 2 compact per-l blocks
};

// Pointer to D_l(edge)[0][0] and its row stride; D_l[row][col] = ptr[row * stride + col].
__device__ __forceinline__ const float* wigner_block(
    const WignerRef& w, int64_t edge, int l, int64_t& row_stride) {
  if (w.mode == 2) {
    row_stride = 2 * l + 1;
    return w.data + edge * w.compact_stride + w.compact_offsets[l];
  }
  const int64_t off = w.offsets[l];
  row_stride = w.dense_stride;
  return w.data + (edge * w.dense_stride + off) * w.dense_stride + off;
}

// Gradient of the multi-m pair pack for one channel of degree L:
// grad_x[base + d] = sum_j D[d][j] g[j], where g[l-m] / g[l+m] are the
// gradients of the (-m, +m) pair entries of the channel in block m.
template <int L>
__device__ __forceinline__ void pack_grad_channel(
    const float* __restrict__ grad_edge,
    int64_t total_cin,
    const int32_t* __restrict__ cols,
    int mtab,
    const WignerRef& w,
    int64_t edge,
    bool rotate,
    float* __restrict__ out) {
  constexpr int dim = 2 * L + 1;
  float g[dim];
#pragma unroll
  for (int j = 0; j < dim; ++j) {
    g[j] = 0.0f;
  }
#pragma unroll
  for (int m = 1; m <= L; ++m) {
    if (m <= mtab) {
      const int col = cols[m - 1];
      if (col >= 0) {
        g[L - m] = grad_edge[col];
        g[L + m] = grad_edge[total_cin + col];
      }
    }
  }
  if (!rotate || w.mode == 0) {
#pragma unroll
    for (int d = 0; d < dim; ++d) {
      out[d] = g[d];
    }
    return;
  }
  int64_t rs;
  const float* __restrict__ D = wigner_block(w, edge, L, rs);
#pragma unroll
  for (int d = 0; d < dim; ++d) {
    float acc = 0.0f;
#pragma unroll
    for (int j = 0; j < dim; ++j) {
      acc = fmaf(__ldg(D + d * rs + j), g[j], acc);
    }
    out[d] = acc;
  }
}

__device__ void pack_grad_channel_generic(
    const float* __restrict__ grad_edge,
    int64_t total_cin,
    const int32_t* __restrict__ cols,
    int mtab,
    const WignerRef& w,
    int64_t edge,
    int l,
    bool rotate,
    float* __restrict__ out) {
  const int dim = 2 * l + 1;
  int64_t rs = 0;
  const float* __restrict__ D = (rotate && w.mode != 0) ? wigner_block(w, edge, l, rs) : nullptr;
  for (int d = 0; d < dim; ++d) {
    float acc = 0.0f;
    for (int m = 1; m <= l && m <= mtab; ++m) {
      const int col = cols[m - 1];
      if (col < 0) {
        continue;
      }
      const float g0 = grad_edge[col];
      const float g1 = grad_edge[total_cin + col];
      if (D == nullptr) {
        acc += (d == l - m ? g0 : 0.0f) + (d == l + m ? g1 : 0.0f);
      } else {
        acc = fmaf(D[d * rs + l - m], g0, acc);
        acc = fmaf(D[d * rs + l + m], g1, acc);
      }
    }
    out[d] = acc;
  }
}

__global__ void channel_pack_grad_kernel(
    const float* __restrict__ grad_packed,
    WignerRef w,
    const int32_t* __restrict__ ch_base,
    const int32_t* __restrict__ ch_l,
    const int32_t* __restrict__ ch_cols,
    float* __restrict__ grad_x,
    int64_t n_edges,
    int64_t n_channels,
    int64_t in_dim,
    int64_t total_cin,
    int mtab,
    bool rotate) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_edges * n_channels) {
    return;
  }
  const int64_t edge = idx / n_channels;
  const int64_t k = idx - edge * n_channels;
  const int l = ch_l[k];
  const float* __restrict__ grad_edge = grad_packed + edge * 2 * total_cin;
  const int32_t* __restrict__ cols = ch_cols + k * mtab;
  float* __restrict__ out = grad_x + edge * in_dim + ch_base[k];
  switch (l) {
    case 0: out[0] = 0.0f; break;
    case 1: pack_grad_channel<1>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 2: pack_grad_channel<2>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 3: pack_grad_channel<3>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 4: pack_grad_channel<4>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 5: pack_grad_channel<5>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 6: pack_grad_channel<6>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 7: pack_grad_channel<7>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    case 8: pack_grad_channel<8>(grad_edge, total_cin, cols, mtab, w, edge, rotate, out); break;
    default: pack_grad_channel_generic(grad_edge, total_cin, cols, mtab, w, edge, l, rotate, out); break;
  }
}

WignerRef make_wigner_ref(
    const torch::Tensor& wigner,
    const torch::Tensor& offsets,
    const torch::Tensor& compact_offsets,
    int64_t wigner_mode,
    int64_t wigner_stride,
    bool rotate) {
  WignerRef w;
  w.mode = rotate ? static_cast<int>(wigner_mode) : 0;
  w.data = (rotate && wigner.numel() > 0) ? wigner.data_ptr<float>() : nullptr;
  w.offsets = offsets.numel() > 0 ? offsets.data_ptr<int64_t>() : nullptr;
  w.compact_offsets = compact_offsets.numel() > 0 ? compact_offsets.data_ptr<int64_t>() : nullptr;
  w.dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  w.compact_stride = wigner_stride;
  if (w.data == nullptr) {
    w.mode = 0;
  }
  return w;
}

}  // namespace

torch::Tensor channel_pack_grad_fp32_cuda(
    torch::Tensor grad_packed,
    torch::Tensor wigner,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor ch_base,
    torch::Tensor ch_l,
    torch::Tensor ch_cols,
    int64_t in_dim,
    bool zero_fill,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_packed.size(0);
  const int64_t total_cin = grad_packed.size(2);
  const int64_t n_channels = ch_base.numel();
  const int mtab = n_channels > 0 ? static_cast<int>(ch_cols.numel() / n_channels) : 0;
  auto grad_x = zero_fill ? torch::zeros({n_edges, in_dim}, grad_packed.options())
                          : torch::empty({n_edges, in_dim}, grad_packed.options());
  if (n_edges == 0 || n_channels == 0) {
    return grad_x;
  }
  const WignerRef w = make_wigner_ref(wigner, offsets, compact_offsets, wigner_mode, wigner_stride, rotate_in);
  const int64_t total = n_edges * n_channels;
  const dim3 grid((total + kChannelThreads - 1) / kChannelThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  channel_pack_grad_kernel<<<grid, kChannelThreads, 0, stream>>>(
      grad_packed.data_ptr<float>(), w,
      ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
      grad_x.data_ptr<float>(), n_edges, n_channels, in_dim, total_cin, mtab, rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_x;
}
