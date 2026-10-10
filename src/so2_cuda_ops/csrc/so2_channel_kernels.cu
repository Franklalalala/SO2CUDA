// Channel-major SO(2) pack / scatter kernels and their block layout.
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

// ---------------------------------------------------------------------------
// Block layout. The pair data of one layer call live in one flat buffer made of
// one block per m: block m holds N rows of [pair 0 (c_m) | pair 1 (c_m)] for
// m > 0 (pair 0 is the -m component, pair 1 the +m component) and N rows of
// c_0 values for m = 0. Row e of block m starts at
//   n_edges * prefix[m] + row(e) * stride[m]
// where prefix[m] is the per-edge width of the blocks before m and stride[m]
// the per-edge width of block m (2 c_m or c_0); row(e) = e unless a row
// permutation is given (rows sorted by routing group).

constexpr int kMaxBlocks = 16;

// Device table of the blocks, three int64 per m: prefix, c_m (0 when absent), stride.
struct BlockSet {
  const int64_t* __restrict__ table;
};

__device__ __forceinline__ int64_t block_row(const BlockSet& b, int m, int64_t n_edges, int64_t row) {
  return n_edges * b.table[3 * m] + row * b.table[3 * m + 2];
}

__device__ __forceinline__ int block_width(const BlockSet& b, int m) {
  return static_cast<int>(b.table[3 * m + 1]);
}

// Rotate one channel of degree L into the blocks: r_j = sum_d v[d] D[d][j],
// r_{L-m} -> pair 0 and r_{L+m} -> pair 1 of block m, r_L -> block 0.
template <int L>
__device__ __forceinline__ void rotate_channel_to_blocks(
    const float* __restrict__ src,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    float* __restrict__ dst,
    int64_t n_edges,
    int64_t row,
    const float* __restrict__ D,
    int64_t rs,
    float scale) {
  constexpr int dim = 2 * L + 1;
  float v[dim];
#pragma unroll
  for (int d = 0; d < dim; ++d) {
    v[d] = src[d];
  }
  if (block_width(blocks, 0) > 0 && cols[0] >= 0) {
    float r = v[L];
    if (D != nullptr && L > 0) {
      r = 0.0f;
#pragma unroll
      for (int d = 0; d < dim; ++d) {
        r = fmaf(v[d], __ldg(D + d * rs + L), r);
      }
    }
    dst[block_row(blocks, 0, n_edges, row) + cols[0]] = r * scale;
  }
#pragma unroll
  for (int m = 1; m <= L; ++m) {
    if (m > mtab) {
      break;
    }
    const int col = cols[m];
    if (col < 0 || block_width(blocks, m) == 0) {
      continue;
    }
    float r0 = v[L - m];
    float r1 = v[L + m];
    if (D != nullptr) {
      r0 = 0.0f;
      r1 = 0.0f;
#pragma unroll
      for (int d = 0; d < dim; ++d) {
        r0 = fmaf(v[d], __ldg(D + d * rs + (L - m)), r0);
        r1 = fmaf(v[d], __ldg(D + d * rs + (L + m)), r1);
      }
    }
    float* __restrict__ out = dst + block_row(blocks, m, n_edges, row) + col;
    out[0] = r0 * scale;
    out[block_width(blocks, m)] = r1 * scale;
  }
}

// Gather one channel of degree L from the blocks and rotate it back:
// out[d] = sum_j D[d][j] g[j] with g_{L-m}, g_{L+m} from block m and g_L from block 0.
template <int L>
__device__ __forceinline__ void gather_channel_from_blocks(
    const float* __restrict__ src,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    int64_t n_edges,
    int64_t row,
    const float* __restrict__ D,
    int64_t rs,
    float scale,
    bool accumulate,
    float* __restrict__ out) {
  constexpr int dim = 2 * L + 1;
  float g[dim];
#pragma unroll
  for (int j = 0; j < dim; ++j) {
    g[j] = 0.0f;
  }
  if (block_width(blocks, 0) > 0 && cols[0] >= 0) {
    g[L] = src[block_row(blocks, 0, n_edges, row) + cols[0]];
  }
#pragma unroll
  for (int m = 1; m <= L; ++m) {
    if (m > mtab) {
      break;
    }
    const int col = cols[m];
    if (col < 0 || block_width(blocks, m) == 0) {
      continue;
    }
    const float* __restrict__ in = src + block_row(blocks, m, n_edges, row) + col;
    g[L - m] = in[0];
    g[L + m] = in[block_width(blocks, m)];
  }
  const int reach = mtab < L ? mtab : L;
#pragma unroll
  for (int d = 0; d < dim; ++d) {
    float acc;
    if (D == nullptr || L == 0) {
      acc = g[d];
    } else {
      acc = 0.0f;
#pragma unroll
      for (int j = 0; j < dim; ++j) {
        if (j - L <= reach && L - j <= reach) {
          acc = fmaf(__ldg(D + d * rs + j), g[j], acc);
        }
      }
    }
    acc *= scale;
    out[d] = accumulate ? out[d] + acc : acc;
  }
}

// Degrees above the unrolled range: same arithmetic without register arrays.
__device__ void rotate_channel_to_blocks_any(
    const float* __restrict__ src,
    int l,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    float* __restrict__ dst,
    int64_t n_edges,
    int64_t row,
    const float* __restrict__ D,
    int64_t rs,
    float scale) {
  const int dim = 2 * l + 1;
  for (int m = 0; m <= l && m <= mtab; ++m) {
    const int col = cols[m];
    if (col < 0 || block_width(blocks, m) == 0) {
      continue;
    }
    float* __restrict__ out = dst + block_row(blocks, m, n_edges, row) + col;
    for (int p = 0; p < (m == 0 ? 1 : 2); ++p) {
      const int j = p == 0 ? l - m : l + m;
      float r = src[j];
      if (D != nullptr) {
        r = 0.0f;
        for (int d = 0; d < dim; ++d) {
          r = fmaf(src[d], D[d * rs + j], r);
        }
      }
      out[p * block_width(blocks, m)] = r * scale;
    }
  }
}

__device__ void gather_channel_from_blocks_any(
    const float* __restrict__ src,
    int l,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    int64_t n_edges,
    int64_t row,
    const float* __restrict__ D,
    int64_t rs,
    float scale,
    bool accumulate,
    float* __restrict__ out) {
  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    float acc = 0.0f;
    for (int m = 0; m <= l && m <= mtab; ++m) {
      const int col = cols[m];
      if (col < 0 || block_width(blocks, m) == 0) {
        continue;
      }
      const float* __restrict__ in = src + block_row(blocks, m, n_edges, row) + col;
      for (int p = 0; p < (m == 0 ? 1 : 2); ++p) {
        const int j = p == 0 ? l - m : l + m;
        const float g = in[p * block_width(blocks, m)];
        acc = D == nullptr ? (d == j ? acc + g : acc) : fmaf(D[d * rs + j], g, acc);
      }
    }
    acc *= scale;
    out[d] = accumulate ? out[d] + acc : acc;
  }
}

#define SO2_DISPATCH_DEGREE(l, CALL, GENERIC)            \
  switch (l) {                                     \
    case 0: { constexpr int LL = 0; CALL; } break; \
    case 1: { constexpr int LL = 1; CALL; } break; \
    case 2: { constexpr int LL = 2; CALL; } break; \
    case 3: { constexpr int LL = 3; CALL; } break; \
    case 4: { constexpr int LL = 4; CALL; } break; \
    case 5: { constexpr int LL = 5; CALL; } break; \
    case 6: { constexpr int LL = 6; CALL; } break; \
    case 7: { constexpr int LL = 7; CALL; } break; \
    case 8: { constexpr int LL = 8; CALL; } break; \
    default: GENERIC; break;                       \
  }

__global__ void channel_rotate_to_blocks_kernel(
    const float* __restrict__ src,
    int64_t src_stride,
    WignerRef w,
    const int32_t* __restrict__ ch_base,
    const int32_t* __restrict__ ch_l,
    const int32_t* __restrict__ ch_cols,
    int cols_per_channel,
    int mtab,
    BlockSet blocks,
    float* __restrict__ dst,
    const float* __restrict__ edge_scale,
    const int64_t* __restrict__ row_of_edge,
    int64_t n_edges,
    int64_t n_channels) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n_edges * n_channels) {
    return;
  }
  const int64_t edge = idx / n_channels;
  const int64_t k = idx - edge * n_channels;
  const int l = ch_l[k];
  const int64_t row = row_of_edge == nullptr ? edge : row_of_edge[edge];
  const float* __restrict__ in = src + edge * src_stride + ch_base[k];
  const int32_t* __restrict__ cols = ch_cols + k * cols_per_channel;
  int64_t rs = 0;
  const float* __restrict__ D = (w.mode != 0 && l > 0) ? wigner_block(w, edge, l, rs) : nullptr;
  const float scale = edge_scale == nullptr ? 1.0f : edge_scale[edge];
  SO2_DISPATCH_DEGREE(l, (rotate_channel_to_blocks<LL>(in, cols, mtab, blocks, dst, n_edges, row, D, rs, scale)),
                      (rotate_channel_to_blocks_any(in, l, cols, mtab, blocks, dst, n_edges, row, D, rs, scale)));
}

// Largest degree whose 2l+1 outputs a warp stages in shared memory.
constexpr int kStagedMaxDegree = 8;
constexpr int kStageWidth = 2 * kStagedMaxDegree + 1;

__global__ void channel_gather_from_blocks_kernel(
    const float* __restrict__ src,
    WignerRef w,
    const int32_t* __restrict__ ch_base,
    const int32_t* __restrict__ ch_l,
    const int32_t* __restrict__ ch_cols,
    int cols_per_channel,
    int mtab,
    BlockSet blocks,
    float* __restrict__ dst,
    int64_t dst_stride,
    const float* __restrict__ edge_scale,
    bool accumulate,
    const int64_t* __restrict__ row_of_edge,
    int64_t n_edges,
    int64_t n_channels) {
  // When the 32 lanes of a warp hold 32 consecutive channels of one degree of one
  // edge (every multiplicity-32 irrep block), their 32*(2l+1) outputs form one
  // contiguous span: the warp stages them in shared memory and writes the span with
  // coalesced stores instead of 2l+1 strided stores per lane.
  __shared__ float stage_all[kChannelThreads * kStageWidth];
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = idx < n_edges * n_channels;
  int64_t edge = 0;
  int64_t k = 0;
  int l = -1;
  int base = 0;
  if (active) {
    edge = idx / n_channels;
    k = idx - edge * n_channels;
    l = ch_l[k];
    base = ch_base[k];
  }
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int l0 = __shfl_sync(full, l, 0);
  const long long e0 = __shfl_sync(full, static_cast<long long>(edge), 0);
  const int b0 = __shfl_sync(full, base, 0);
  const bool fits = active && l == l0 && static_cast<long long>(edge) == e0 && l <= kStagedMaxDegree &&
                    base == b0 + lane * (2 * l + 1);
  const bool staged = __all_sync(full, fits);
  if (!active) {
    return;
  }
  const int64_t row = row_of_edge == nullptr ? edge : row_of_edge[edge];
  const int32_t* __restrict__ cols = ch_cols + k * cols_per_channel;
  int64_t rs = 0;
  const float* __restrict__ D = (w.mode != 0 && l > 0) ? wigner_block(w, edge, l, rs) : nullptr;
  const float scale = edge_scale == nullptr ? 1.0f : edge_scale[edge];
  if (staged) {
    float* __restrict__ stage = stage_all + (threadIdx.x - lane) * kStageWidth;
    float* __restrict__ mine = stage + lane * (2 * l + 1);
    SO2_DISPATCH_DEGREE(l, (gather_channel_from_blocks<LL>(src, cols, mtab, blocks, n_edges, row, D, rs, scale,
                                                           false, mine)),
                        ((void)0));
    __syncwarp();
    const int span = 32 * (2 * l + 1);
    float* __restrict__ out = dst + edge * dst_stride + b0;
    if (accumulate) {
      for (int i = lane; i < span; i += 32) {
        out[i] += stage[i];
      }
    } else {
      for (int i = lane; i < span; i += 32) {
        out[i] = stage[i];
      }
    }
    return;
  }
  float* __restrict__ out = dst + edge * dst_stride + base;
  SO2_DISPATCH_DEGREE(l, (gather_channel_from_blocks<LL>(src, cols, mtab, blocks, n_edges, row, D, rs, scale,
                                                         accumulate, out)),
                      (gather_channel_from_blocks_any(src, l, cols, mtab, blocks, n_edges, row, D, rs, scale,
                                                      accumulate, out)));
}

// [[A, -B], [B, A]] for every m block from the stacked [A; B] pair weights.
// Each block holds `groups` stacked weights (one per routing group).
struct WeightSet {
  const float* src[kMaxBlocks];
  float* dst[kMaxBlocks];
  int32_t cout[kMaxBlocks];
  int32_t cin[kMaxBlocks];
  int32_t groups[kMaxBlocks];
  int64_t prefix[kMaxBlocks + 1];  // running count of written entries
  int count;
};

__global__ void block_complex_weights_kernel(WeightSet ws, bool backward) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= ws.prefix[ws.count]) {
    return;
  }
  int b = 0;
  while (b + 1 < ws.count && idx >= ws.prefix[b + 1]) {
    ++b;
  }
  const int64_t local = idx - ws.prefix[b];
  const int co = ws.cout[b];
  const int ci = ws.cin[b];
  if (!backward) {
    // forward: local indexes [groups, 2co, 2ci] block-complex matrices
    const int64_t per = 4 * static_cast<int64_t>(co) * ci;
    const int64_t g = local / per;
    const int64_t rem = local - g * per;
    const int row = static_cast<int>(rem / (2 * ci));
    const int col = static_cast<int>(rem - static_cast<int64_t>(row) * 2 * ci);
    const int o = row < co ? row : row - co;
    const int i = col < ci ? col : col - ci;
    const float* src = ws.src[b] + g * 2 * static_cast<int64_t>(co) * ci;
    const float a = src[static_cast<int64_t>(o) * ci + i];
    const float bb = src[static_cast<int64_t>(co + o) * ci + i];
    float value;
    if (row < co) {
      value = col < ci ? a : -bb;
    } else {
      value = col < ci ? bb : a;
    }
    ws.dst[b][local] = value;
  } else {
    // backward: local indexes [groups, 2co, ci] stacked gradients from [groups, 2co, 2ci] ones
    const int64_t per = 2 * static_cast<int64_t>(co) * ci;
    const int64_t grp = local / per;
    const int64_t rem = local - grp * per;
    const int row = static_cast<int>(rem / ci);
    const int i = static_cast<int>(rem - static_cast<int64_t>(row) * ci);
    const float* g = ws.src[b] + grp * 4 * static_cast<int64_t>(co) * ci;
    const int64_t w2 = 2 * ci;
    float value;
    if (row < co) {  // d/dA = G11 + G22
      value = g[row * w2 + i] + g[(co + row) * w2 + ci + i];
    } else {         // d/dB = G21 - G12
      const int o = row - co;
      value = g[(co + o) * w2 + i] - g[o * w2 + ci + i];
    }
    ws.dst[b][local] = value;
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

namespace {

BlockSet make_block_set(const torch::Tensor& block_table) {
  BlockSet b;
  b.table = block_table.data_ptr<int64_t>();
  return b;
}

}  // namespace

torch::Tensor channel_rotate_to_blocks_fp32_cuda(
    torch::Tensor src,
    torch::Tensor wigner,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor ch_base,
    torch::Tensor ch_l,
    torch::Tensor ch_cols,
    torch::Tensor block_table,
    int64_t total_width,
    torch::Tensor edge_scale,
    torch::Tensor row_of_edge,
    bool rotate,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = src.size(0);
  const int64_t n_channels = ch_base.numel();
  auto dst = torch::empty({n_edges * total_width}, src.options());
  if (n_edges == 0 || n_channels == 0 || total_width == 0) {
    return dst;
  }
  const int cols_per_channel = static_cast<int>(ch_cols.numel() / n_channels);
  const BlockSet blocks = make_block_set(block_table);
  const WignerRef w = make_wigner_ref(wigner, offsets, compact_offsets, wigner_mode, wigner_stride, rotate);
  const int64_t total = n_edges * n_channels;
  const dim3 grid((total + kChannelThreads - 1) / kChannelThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  channel_rotate_to_blocks_kernel<<<grid, kChannelThreads, 0, stream>>>(
      src.data_ptr<float>(), src.size(1), w,
      ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
      cols_per_channel, cols_per_channel - 1, blocks, dst.data_ptr<float>(),
      edge_scale.numel() > 0 ? edge_scale.data_ptr<float>() : nullptr,
      row_of_edge.numel() > 0 ? row_of_edge.data_ptr<int64_t>() : nullptr,
      n_edges, n_channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

torch::Tensor channel_gather_from_blocks_fp32_cuda(
    torch::Tensor src,
    int64_t n_edges,
    torch::Tensor wigner,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor ch_base,
    torch::Tensor ch_l,
    torch::Tensor ch_cols,
    torch::Tensor block_table,
    int64_t dst_dim,
    bool zero_fill,
    torch::Tensor edge_scale,
    torch::Tensor accumulate_into,
    torch::Tensor row_of_edge,
    bool rotate,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_channels = ch_base.numel();
  const bool accumulate = accumulate_into.numel() > 0;
  torch::Tensor dst;
  if (accumulate) {
    dst = accumulate_into;
  } else {
    dst = zero_fill ? torch::zeros({n_edges, dst_dim}, src.options())
                    : torch::empty({n_edges, dst_dim}, src.options());
  }
  if (n_edges == 0 || n_channels == 0) {
    return dst;
  }
  const int cols_per_channel = static_cast<int>(ch_cols.numel() / n_channels);
  const BlockSet blocks = make_block_set(block_table);
  const WignerRef w = make_wigner_ref(wigner, offsets, compact_offsets, wigner_mode, wigner_stride, rotate);
  const int64_t total = n_edges * n_channels;
  const dim3 grid((total + kChannelThreads - 1) / kChannelThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  channel_gather_from_blocks_kernel<<<grid, kChannelThreads, 0, stream>>>(
      src.data_ptr<float>(), w,
      ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
      cols_per_channel, cols_per_channel - 1, blocks, dst.data_ptr<float>(), dst.size(1),
      edge_scale.numel() > 0 ? edge_scale.data_ptr<float>() : nullptr, accumulate,
      row_of_edge.numel() > 0 ? row_of_edge.data_ptr<int64_t>() : nullptr,
      n_edges, n_channels);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return dst;
}

torch::Tensor block_complex_weights_fp32_cuda(std::vector<torch::Tensor> weights) {
  TORCH_CHECK(!weights.empty() && static_cast<int>(weights.size()) <= kMaxBlocks,
              "between 1 and ", kMaxBlocks, " pair weights are required");
  WeightSet ws;
  ws.count = static_cast<int>(weights.size());
  ws.prefix[0] = 0;
  for (int b = 0; b < ws.count; ++b) {
    const auto& weight = weights[b];
    TORCH_CHECK((weight.dim() == 2 || weight.dim() == 3) && weight.size(-2) % 2 == 0,
                "pair weights must be [2*Cout, Cin] or [groups, 2*Cout, Cin]");
    ws.groups[b] = static_cast<int32_t>(weight.dim() == 3 ? weight.size(0) : 1);
    ws.cout[b] = static_cast<int32_t>(weight.size(-2) / 2);
    ws.cin[b] = static_cast<int32_t>(weight.size(-1));
    ws.prefix[b + 1] = ws.prefix[b] + 4 * static_cast<int64_t>(ws.groups[b]) * ws.cout[b] * ws.cin[b];
  }
  auto out = torch::empty({ws.prefix[ws.count]}, weights[0].options());
  for (int b = 0; b < ws.count; ++b) {
    ws.src[b] = weights[b].data_ptr<float>();
    ws.dst[b] = out.data_ptr<float>() + ws.prefix[b];
  }
  if (ws.prefix[ws.count] == 0) {
    return out;
  }
  const int threads = 256;
  const dim3 grid((ws.prefix[ws.count] + threads - 1) / threads);
  block_complex_weights_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(ws, false);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> block_complex_weight_grads_fp32_cuda(
    torch::Tensor grad_flat,
    std::vector<int64_t> couts,
    std::vector<int64_t> cins,
    int64_t groups) {
  TORCH_CHECK(couts.size() == cins.size() && !couts.empty() && static_cast<int>(couts.size()) <= kMaxBlocks,
              "between 1 and ", kMaxBlocks, " blocks are required");
  TORCH_CHECK(groups >= 0, "groups must be nonnegative (0 = ungrouped)");
  WeightSet ws;
  ws.count = static_cast<int>(couts.size());
  std::vector<torch::Tensor> grads;
  grads.reserve(couts.size());
  int64_t src_offset = 0;
  const int64_t g = groups > 0 ? groups : 1;
  ws.prefix[0] = 0;
  for (int b = 0; b < ws.count; ++b) {
    ws.cout[b] = static_cast<int32_t>(couts[b]);
    ws.cin[b] = static_cast<int32_t>(cins[b]);
    ws.groups[b] = static_cast<int32_t>(g);
    grads.push_back(groups > 0 ? torch::empty({groups, 2 * couts[b], cins[b]}, grad_flat.options())
                               : torch::empty({2 * couts[b], cins[b]}, grad_flat.options()));
    ws.src[b] = grad_flat.data_ptr<float>() + src_offset;
    ws.dst[b] = grads.back().data_ptr<float>();
    src_offset += 4 * g * couts[b] * cins[b];
    ws.prefix[b + 1] = ws.prefix[b] + 2 * g * couts[b] * cins[b];
  }
  TORCH_CHECK(src_offset == grad_flat.numel(), "block-complex gradient size mismatch");
  if (ws.prefix[ws.count] == 0) {
    return grads;
  }
  const int threads = 256;
  const dim3 grid((ws.prefix[ws.count] + threads - 1) / threads);
  block_complex_weights_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(ws, true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grads;
}
