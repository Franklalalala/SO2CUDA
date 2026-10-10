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

// Copies of the block buffer: copy c (one per routing slot) starts at c * copy_stride
// and stores edge e in row rows[c * n_edges + e] (row e when rows is null), scaled by
// scales[c * n_edges + e] (1 when scales is null).
struct Copies {
  const int64_t* __restrict__ rows;
  const float* __restrict__ scales;
  int64_t stride;
  int count;
};

__device__ __forceinline__ int64_t copy_row(const Copies& c, int i, int64_t n_edges, int64_t edge) {
  return c.rows == nullptr ? edge : c.rows[i * n_edges + edge];
}

__device__ __forceinline__ float copy_scale(const Copies& c, int i, int64_t n_edges, int64_t edge) {
  return c.scales == nullptr ? 1.0f : c.scales[i * n_edges + edge];
}

// Per-block radial weights of the input side: column c of block m is scaled by
// ptr[m][edge * stride[m] + c] (no scaling when ptr[m] is null). `plain`, when set,
// receives the unscaled rotated values in edge order (one more block buffer).
struct RadialSet {
  const float* ptr[kMaxBlocks];
  int64_t stride[kMaxBlocks];
  float* plain;
};

__device__ __forceinline__ float radial_value(const RadialSet& r, int m, int64_t edge, int col) {
  const float* p = r.ptr[m];
  return p == nullptr ? 1.0f : p[edge * r.stride[m] + col];
}

// Rotate one channel of degree L into the blocks of every copy: r_j = sum_d v[d] D[d][j],
// r_{L-m} -> pair 0 and r_{L+m} -> pair 1 of block m, r_L -> block 0. The rotation is
// computed once and written to each copy.
template <int L>
__device__ __forceinline__ void rotate_channel_to_blocks(
    const float* src,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    float* __restrict__ dst,
    const Copies& copies,
    const RadialSet& radial,
    int64_t n_edges,
    int64_t edge,
    const float* __restrict__ D,
    int64_t rs) {
  constexpr int dim = 2 * L + 1;
  const int reach = mtab < L ? mtab : L;
  float v[dim];
#pragma unroll
  for (int d = 0; d < dim; ++d) {
    v[d] = src[d];
  }
  float r[dim];
#pragma unroll
  for (int j = 0; j < dim; ++j) {
    r[j] = 0.0f;
    if (j - L <= reach && L - j <= reach) {
      if (D == nullptr || L == 0) {
        r[j] = v[j];
      } else {
#pragma unroll
        for (int d = 0; d < dim; ++d) {
          r[j] = fmaf(v[d], D[d * rs + j], r[j]);
        }
      }
    }
  }
  if (radial.plain != nullptr) {
    if (block_width(blocks, 0) > 0 && cols[0] >= 0) {
      radial.plain[block_row(blocks, 0, n_edges, edge) + cols[0]] = r[L];
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
      float* __restrict__ out = radial.plain + block_row(blocks, m, n_edges, edge) + col;
      out[0] = r[L - m];
      out[block_width(blocks, m)] = r[L + m];
    }
  }
  float w[L + 1];
#pragma unroll
  for (int m = 0; m <= L; ++m) {
    w[m] = (m <= mtab && cols[m] >= 0) ? radial_value(radial, m, edge, cols[m]) : 1.0f;
  }
  for (int c = 0; c < copies.count; ++c) {
    const int64_t row = copy_row(copies, c, n_edges, edge);
    const float scale = copy_scale(copies, c, n_edges, edge);
    float* __restrict__ out_copy = dst + c * copies.stride;
    if (block_width(blocks, 0) > 0 && cols[0] >= 0) {
      out_copy[block_row(blocks, 0, n_edges, row) + cols[0]] = r[L] * (scale * w[0]);
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
      float* __restrict__ out = out_copy + block_row(blocks, m, n_edges, row) + col;
      out[0] = r[L - m] * (scale * w[m]);
      out[block_width(blocks, m)] = r[L + m] * (scale * w[m]);
    }
  }
}

// Gather one channel of degree L from the blocks of every copy and rotate it back:
// g = sum_c scale_c * (g_{L-m}, g_{L+m} from block m, g_L from block 0 of copy c) and
// out[d] = sum_j D[d][j] g[j].
template <int L>
__device__ __forceinline__ void gather_channel_from_blocks(
    const float* __restrict__ src,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    const Copies& copies,
    int64_t n_edges,
    int64_t edge,
    const float* __restrict__ D,
    int64_t rs,
    bool accumulate,
    float* __restrict__ out) {
  constexpr int dim = 2 * L + 1;
  float g[dim];
#pragma unroll
  for (int j = 0; j < dim; ++j) {
    g[j] = 0.0f;
  }
  for (int c = 0; c < copies.count; ++c) {
    const int64_t row = copy_row(copies, c, n_edges, edge);
    const float scale = copy_scale(copies, c, n_edges, edge);
    const float* __restrict__ in_copy = src + c * copies.stride;
    if (block_width(blocks, 0) > 0 && cols[0] >= 0) {
      g[L] = fmaf(scale, in_copy[block_row(blocks, 0, n_edges, row) + cols[0]], g[L]);
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
      const float* __restrict__ in = in_copy + block_row(blocks, m, n_edges, row) + col;
      g[L - m] = fmaf(scale, in[0], g[L - m]);
      g[L + m] = fmaf(scale, in[block_width(blocks, m)], g[L + m]);
    }
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
          acc = fmaf(D[d * rs + j], g[j], acc);
        }
      }
    }
    out[d] = accumulate ? out[d] + acc : acc;
  }
}

// Degrees above the unrolled range: same arithmetic without register arrays.
__device__ void rotate_channel_to_blocks_any(
    const float* src,
    int l,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    float* __restrict__ dst,
    const Copies& copies,
    const RadialSet& radial,
    int64_t n_edges,
    int64_t edge,
    const float* __restrict__ D,
    int64_t rs) {
  const int dim = 2 * l + 1;
  for (int m = 0; m <= l && m <= mtab; ++m) {
    const int col = cols[m];
    if (col < 0 || block_width(blocks, m) == 0) {
      continue;
    }
    const float w = radial_value(radial, m, edge, col);
    for (int p = 0; p < (m == 0 ? 1 : 2); ++p) {
      const int j = p == 0 ? l - m : l + m;
      float r = src[j];
      if (D != nullptr) {
        r = 0.0f;
        for (int d = 0; d < dim; ++d) {
          r = fmaf(src[d], D[d * rs + j], r);
        }
      }
      if (radial.plain != nullptr) {
        radial.plain[block_row(blocks, m, n_edges, edge) + col + p * block_width(blocks, m)] = r;
      }
      for (int c = 0; c < copies.count; ++c) {
        float* __restrict__ out = dst + c * copies.stride +
                                  block_row(blocks, m, n_edges, copy_row(copies, c, n_edges, edge)) + col;
        out[p * block_width(blocks, m)] = r * (copy_scale(copies, c, n_edges, edge) * w);
      }
    }
  }
}

__device__ void gather_channel_from_blocks_any(
    const float* __restrict__ src,
    int l,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    const Copies& copies,
    int64_t n_edges,
    int64_t edge,
    const float* __restrict__ D,
    int64_t rs,
    bool accumulate,
    float* __restrict__ out) {
  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    float acc = 0.0f;
    for (int c = 0; c < copies.count; ++c) {
      const int64_t row = copy_row(copies, c, n_edges, edge);
      const float scale = copy_scale(copies, c, n_edges, edge);
      for (int m = 0; m <= l && m <= mtab; ++m) {
        const int col = cols[m];
        if (col < 0 || block_width(blocks, m) == 0) {
          continue;
        }
        const float* __restrict__ in = src + c * copies.stride + block_row(blocks, m, n_edges, row) + col;
        for (int p = 0; p < (m == 0 ? 1 : 2); ++p) {
          const int j = p == 0 ? l - m : l + m;
          const float g = scale * in[p * block_width(blocks, m)];
          acc = D == nullptr ? (d == j ? acc + g : acc) : fmaf(D[d * rs + j], g, acc);
        }
      }
    }
    out[d] = accumulate ? out[d] + acc : acc;
  }
}

// Largest degree whose 2l+1 coefficients a warp stages in shared memory.
constexpr int kStagedMaxDegree = 8;
constexpr int kStageWidth = 2 * kStagedMaxDegree + 1;

// The 32 lanes of a warp hold consecutive channels; when they belong to one edge and
// their feature spans follow each other (consecutive irrep channels, any mix of
// degrees up to kStagedMaxDegree), the warp's 2l+1 coefficients form one contiguous
// span of at most 32 * kStageWidth floats. Returns whether the whole warp qualifies;
// `offset` is the lane's start inside the span and `span` its total length.
__device__ __forceinline__ bool warp_contiguous_span(
    bool active, int64_t edge, int l, int base, int& offset, int& span, int& base0) {
  const unsigned full = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int dim = active ? 2 * l + 1 : 0;
  int scan = dim;
#pragma unroll
  for (int step = 1; step < 32; step <<= 1) {
    const int v = __shfl_up_sync(full, scan, step);
    if (lane >= step) {
      scan += v;
    }
  }
  offset = scan - dim;
  span = __shfl_sync(full, scan, 31);
  base0 = __shfl_sync(full, base, 0);
  const long long e0 = __shfl_sync(full, static_cast<long long>(edge), 0);
  const bool fits = active && static_cast<long long>(edge) == e0 && l <= kStagedMaxDegree && base == base0 + offset;
  return __all_sync(full, fits);
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
    Copies copies,
    RadialSet radial,
    int64_t n_edges,
    int64_t n_channels) {
  __shared__ float stage_all[kChannelThreads * kStageWidth];
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const bool active = idx < n_edges * n_channels;
  int64_t edge = 0;
  int64_t k = 0;
  int l = 0;
  int base = 0;
  if (active) {
    edge = idx / n_channels;
    k = idx - edge * n_channels;
    l = ch_l[k];
    base = ch_base[k];
  }
  int offset = 0;
  int span = 0;
  int base0 = 0;
  const bool staged = warp_contiguous_span(active, edge, l, base, offset, span, base0);
  const float* in = src + edge * src_stride + base;
  if (staged) {
    // One coalesced read of the warp's span instead of 2l+1 strided reads per lane.
    float* __restrict__ stage = stage_all + (threadIdx.x - (threadIdx.x & 31)) * kStageWidth;
    const float* __restrict__ span_src = src + edge * src_stride + base0;
    for (int i = threadIdx.x & 31; i < span; i += 32) {
      stage[i] = span_src[i];
    }
    __syncwarp();
    in = stage + offset;
  }
  if (!active) {
    return;
  }
  const int32_t* __restrict__ cols = ch_cols + k * cols_per_channel;
  int64_t rs = 0;
  const float* __restrict__ D = (w.mode != 0 && l > 0) ? wigner_block(w, edge, l, rs) : nullptr;
  SO2_DISPATCH_DEGREE(l, (rotate_channel_to_blocks<LL>(in, cols, mtab, blocks, dst, copies, radial, n_edges, edge, D,
                                                       rs)),
                      (rotate_channel_to_blocks_any(in, l, cols, mtab, blocks, dst, copies, radial, n_edges, edge, D,
                                                    rs)));
}

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
    Copies copies,
    bool accumulate,
    int64_t n_edges,
    int64_t n_channels) {
  // When the 32 lanes of a warp hold consecutive channels of one edge, their outputs
  // form one contiguous span: the warp stages them in shared memory and writes the
  // span with coalesced stores instead of 2l+1 strided stores per lane.
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
  const int lane = threadIdx.x & 31;
  int offset = 0;
  int span = 0;
  int b0 = 0;
  const bool staged = warp_contiguous_span(active, edge, l, base, offset, span, b0);
  if (!active) {
    return;
  }
  const int32_t* __restrict__ cols = ch_cols + k * cols_per_channel;
  int64_t rs = 0;
  const float* __restrict__ D = (w.mode != 0 && l > 0) ? wigner_block(w, edge, l, rs) : nullptr;
  if (staged) {
    float* __restrict__ stage = stage_all + (threadIdx.x - lane) * kStageWidth;
    float* __restrict__ mine = stage + offset;
    SO2_DISPATCH_DEGREE(l, (gather_channel_from_blocks<LL>(src, cols, mtab, blocks, copies, n_edges, edge, D, rs,
                                                           false, mine)),
                        ((void)0));
    __syncwarp();
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
  SO2_DISPATCH_DEGREE(l, (gather_channel_from_blocks<LL>(src, cols, mtab, blocks, copies, n_edges, edge, D, rs,
                                                         accumulate, out)),
                      (gather_channel_from_blocks_any(src, l, cols, mtab, blocks, copies, n_edges, edge, D, rs,
                                                      accumulate, out)));
}

// ---------------------------------------------------------------------------
// Edge-tiled kernels: one thread block per edge. The edge's input row (rotation) or
// its block rows summed over the copies (gather), its compact Wigner blocks and the
// gathered output row live in shared memory, so every global access is a contiguous
// row segment read or written by consecutive threads. The threads loop over the
// channels of the edge; the per-channel arithmetic is the one of the kernels above.

// Gather one channel of degree L from the staged block rows of one edge (block m of
// the edge starts at table[3m] inside `rows`) and rotate it back into out[0..2L].
template <int L>
__device__ __forceinline__ void gather_channel_tile(
    const float* rows,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    const float* D,
    float* out) {
  constexpr int dim = 2 * L + 1;
  float g[dim];
#pragma unroll
  for (int j = 0; j < dim; ++j) {
    g[j] = 0.0f;
  }
  if (block_width(blocks, 0) > 0 && cols[0] >= 0) {
    g[L] = rows[blocks.table[0] + cols[0]];
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
    const float* in = rows + blocks.table[3 * m] + col;
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
          acc = fmaf(D[d * dim + j], g[j], acc);
        }
      }
    }
    out[d] = acc;
  }
}

__device__ void gather_channel_tile_any(
    const float* rows,
    int l,
    const int32_t* __restrict__ cols,
    int mtab,
    const BlockSet& blocks,
    const float* D,
    float* out) {
  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    float acc = 0.0f;
    for (int m = 0; m <= l && m <= mtab; ++m) {
      const int col = cols[m];
      if (col < 0 || block_width(blocks, m) == 0) {
        continue;
      }
      const float* in = rows + blocks.table[3 * m] + col;
      for (int p = 0; p < (m == 0 ? 1 : 2); ++p) {
        const int j = p == 0 ? l - m : l + m;
        const float g = in[p * block_width(blocks, m)];
        acc = D == nullptr ? (d == j ? acc + g : acc) : fmaf(D[d * dim + j], g, acc);
      }
    }
    out[d] = acc;
  }
}

__device__ __forceinline__ void stage_wigner_row(const WignerRef& w, int64_t edge, int floats, float* sw) {
  const float* __restrict__ row = w.data + edge * w.compact_stride;
  for (int i = threadIdx.x; i < floats; i += blockDim.x) {
    sw[i] = row[i];
  }
}

__global__ void channel_rotate_edge_kernel(
    const float* __restrict__ src,
    int64_t src_stride,
    int src_dim,
    WignerRef w,
    int wigner_floats,
    const int32_t* __restrict__ ch_base,
    const int32_t* __restrict__ ch_l,
    const int32_t* __restrict__ ch_cols,
    int cols_per_channel,
    int mtab,
    BlockSet blocks,
    float* __restrict__ dst,
    Copies copies,
    RadialSet radial,
    int64_t n_edges,
    int n_channels) {
  extern __shared__ float smem[];
  float* sx = smem;
  float* sw = smem + src_dim;
  const int64_t edge = blockIdx.x;
  const float* __restrict__ row = src + edge * src_stride;
  for (int i = threadIdx.x; i < src_dim; i += blockDim.x) {
    sx[i] = row[i];
  }
  if (wigner_floats > 0) {
    stage_wigner_row(w, edge, wigner_floats, sw);
  }
  __syncthreads();
  for (int k = threadIdx.x; k < n_channels; k += blockDim.x) {
    const int l = ch_l[k];
    const float* v = sx + ch_base[k];
    const int32_t* __restrict__ cols = ch_cols + k * cols_per_channel;
    const float* D = (wigner_floats > 0 && l > 0) ? sw + w.compact_offsets[l] : nullptr;
    SO2_DISPATCH_DEGREE(l, (rotate_channel_to_blocks<LL>(v, cols, mtab, blocks, dst, copies, radial, n_edges, edge,
                                                         D, 2 * LL + 1)),
                        (rotate_channel_to_blocks_any(v, l, cols, mtab, blocks, dst, copies, radial, n_edges, edge, D,
                                                      2 * l + 1)));
  }
}

// Extras of the gather. The staged sum S over the copies is scaled by the radial weights
// before the rotation back. In the backward of a front-radial layer S is the gradient of
// the radial-scaled blocks and grad[e, offset[m] + c] = sum over the pair halves of
// S * plain (the unscaled rotated input). In the forward of a back-radial layer sum_out
// receives S before the scaling (edge-order block rows, for the radial gradients).
// dot_out[c * N + e] = <copy c row of src, copy c row of dot_src> (the gate gradients).
struct GatherExtras {
  RadialSet radial;
  bool scale_radial;
  float* radial_grad;
  int64_t radial_grad_stride;
  int radial_offset[kMaxBlocks];
  const float* dot_src;
  float* dot_out;
  float* sum_out;
};

__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const unsigned full = 0xffffffffu;
#pragma unroll
  for (int step = 16; step > 0; step >>= 1) {
    v += __shfl_down_sync(full, v, step);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    scratch[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  if (threadIdx.x == 0) {
    for (int i = 0; i < static_cast<int>((blockDim.x + 31) / 32); ++i) {
      total += scratch[i];
    }
  }
  __syncthreads();
  return total;
}

__global__ void channel_gather_edge_kernel(
    const float* __restrict__ src,
    WignerRef w,
    int wigner_floats,
    const int32_t* __restrict__ ch_base,
    const int32_t* __restrict__ ch_l,
    const int32_t* __restrict__ ch_cols,
    int cols_per_channel,
    int mtab,
    BlockSet blocks,
    int n_blocks,
    int total_width,
    float* __restrict__ dst,
    int64_t dst_stride,
    int dst_dim,
    bool zero_fill,
    Copies copies,
    GatherExtras extras,
    bool accumulate,
    int64_t n_edges,
    int n_channels) {
  extern __shared__ float smem[];
  __shared__ float scratch[32];
  float* sg = smem;
  float* sw = sg + total_width;
  float* so = sw + wigner_floats;
  const int64_t edge = blockIdx.x;
  for (int i = threadIdx.x; i < total_width; i += blockDim.x) {
    sg[i] = 0.0f;
  }
  __syncthreads();
  for (int c = 0; c < copies.count; ++c) {
    const int64_t row = copy_row(copies, c, n_edges, edge);
    const float scale = copy_scale(copies, c, n_edges, edge);
    const float* __restrict__ in_copy = src + c * copies.stride;
    const float* __restrict__ dot_copy = extras.dot_src == nullptr ? nullptr : extras.dot_src + c * copies.stride;
    float dot = 0.0f;
    for (int m = 0; m < n_blocks; ++m) {
      const int64_t prefix = blocks.table[3 * m];
      const int stride = static_cast<int>(blocks.table[3 * m + 2]);
      const int64_t offset = n_edges * prefix + row * stride;
      const float* __restrict__ seg = in_copy + offset;
      if (dot_copy != nullptr) {
        const float* __restrict__ xseg = dot_copy + offset;
        for (int i = threadIdx.x; i < stride; i += blockDim.x) {
          const float v = seg[i];
          dot = fmaf(v, xseg[i], dot);
          sg[prefix + i] = fmaf(scale, v, sg[prefix + i]);
        }
      } else {
        for (int i = threadIdx.x; i < stride; i += blockDim.x) {
          sg[prefix + i] = fmaf(scale, seg[i], sg[prefix + i]);
        }
      }
    }
    if (dot_copy != nullptr) {
      const float total = block_sum(dot, scratch);
      if (threadIdx.x == 0) {
        extras.dot_out[c * n_edges + edge] = total;
      }
    }
  }
  if (wigner_floats > 0) {
    stage_wigner_row(w, edge, wigner_floats, sw);
  }
  if (zero_fill) {
    for (int i = threadIdx.x; i < dst_dim; i += blockDim.x) {
      so[i] = 0.0f;
    }
  }
  __syncthreads();
  if (extras.scale_radial) {
    // Radial gradients from the unscaled rotated input, then the radial scaling of S.
    for (int m = 0; m < n_blocks; ++m) {
      const int width = block_width(blocks, m);
      if (width == 0) {
        continue;
      }
      const int64_t prefix = blocks.table[3 * m];
      const int stride = static_cast<int>(blocks.table[3 * m + 2]);
      const float* __restrict__ u = extras.radial.plain == nullptr ? nullptr
                                    : extras.radial.plain + n_edges * prefix + edge * stride;
      float* __restrict__ keep = extras.sum_out == nullptr ? nullptr
                                 : extras.sum_out + n_edges * prefix + edge * stride;
      for (int c = threadIdx.x; c < width; c += blockDim.x) {
        const float r = radial_value(extras.radial, m, edge, c);
        float* __restrict__ s0 = sg + prefix + c;
        if (keep != nullptr) {
          keep[c] = s0[0];
          if (m > 0) {
            keep[c + width] = s0[width];
          }
        }
        if (extras.radial_grad != nullptr && u != nullptr) {
          float g = s0[0] * u[c];
          if (m > 0) {
            g = fmaf(s0[width], u[c + width], g);
          }
          extras.radial_grad[edge * extras.radial_grad_stride + extras.radial_offset[m] + c] = g;
        }
        s0[0] *= r;
        if (m > 0) {
          s0[width] *= r;
        }
      }
    }
    __syncthreads();
  }
  for (int k = threadIdx.x; k < n_channels; k += blockDim.x) {
    const int l = ch_l[k];
    const int32_t* __restrict__ cols = ch_cols + k * cols_per_channel;
    const float* D = (wigner_floats > 0 && l > 0) ? sw + w.compact_offsets[l] : nullptr;
    float* out = so + ch_base[k];
    SO2_DISPATCH_DEGREE(l, (gather_channel_tile<LL>(sg, cols, mtab, blocks, D, out)),
                        (gather_channel_tile_any(sg, l, cols, mtab, blocks, D, out)));
  }
  __syncthreads();
  float* __restrict__ out_row = dst + edge * dst_stride;
  if (accumulate) {
    for (int i = threadIdx.x; i < dst_dim; i += blockDim.x) {
      out_row[i] += so[i];
    }
  } else {
    for (int i = threadIdx.x; i < dst_dim; i += blockDim.x) {
      out_row[i] = so[i];
    }
  }
}

// Threads per edge block: one warp per 32 channels, between 64 and 256.
int edge_block_threads(int64_t n_channels) {
  int64_t t = ((n_channels + 31) / 32) * 32;
  return static_cast<int>(t < 64 ? 64 : (t > 256 ? 256 : t));
}

// Allows `bytes` of dynamic shared memory for `kernel`; false when the device cannot.
template <typename Kernel>
bool allow_shared_memory(Kernel kernel, int64_t bytes) {
  if (bytes <= 48 * 1024) {
    return true;
  }
  int device = 0;
  int optin = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (bytes > optin) {
    return false;
  }
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
  return true;
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

RadialSet make_radial_set(const std::vector<torch::Tensor>& radials, float* plain) {
  RadialSet r;
  for (int m = 0; m < kMaxBlocks; ++m) {
    const bool given = m < static_cast<int>(radials.size()) && radials[m].defined() && radials[m].numel() > 0;
    r.ptr[m] = given ? radials[m].data_ptr<float>() : nullptr;
    r.stride[m] = given ? radials[m].stride(0) : 0;
  }
  r.plain = plain;
  return r;
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
    int64_t wigner_stride,
    int64_t copies,
    std::vector<torch::Tensor> radials,
    bool plain) {
  const int64_t n_edges = src.size(0);
  const int64_t n_channels = ch_base.numel();
  auto dst = torch::empty({(copies + (plain ? 1 : 0)) * n_edges * total_width}, src.options());
  if (n_edges == 0 || n_channels == 0 || total_width == 0) {
    return dst;
  }
  const int cols_per_channel = static_cast<int>(ch_cols.numel() / n_channels);
  const BlockSet blocks = make_block_set(block_table);
  const WignerRef w = make_wigner_ref(wigner, offsets, compact_offsets, wigner_mode, wigner_stride, rotate);
  const int64_t total = n_edges * n_channels;
  const dim3 grid((total + kChannelThreads - 1) / kChannelThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const Copies copy_set{row_of_edge.numel() > 0 ? row_of_edge.data_ptr<int64_t>() : nullptr,
                        edge_scale.numel() > 0 ? edge_scale.data_ptr<float>() : nullptr,
                        n_edges * total_width, static_cast<int>(copies)};
  const RadialSet radial_set =
      make_radial_set(radials, plain ? dst.data_ptr<float>() + copies * n_edges * total_width : nullptr);
  // Edge-tiled rotation for identity or compact Wigner data that fit in shared memory.
  const int wigner_floats = w.mode == 2 ? static_cast<int>(w.compact_stride) : 0;
  const int64_t edge_bytes = (src.size(1) + wigner_floats) * static_cast<int64_t>(sizeof(float));
  if ((w.mode == 0 || w.mode == 2) && n_edges <= INT32_MAX && n_channels <= INT32_MAX &&
      allow_shared_memory(channel_rotate_edge_kernel, edge_bytes)) {
    channel_rotate_edge_kernel<<<static_cast<unsigned int>(n_edges), edge_block_threads(n_channels), edge_bytes,
                                 stream>>>(
        src.data_ptr<float>(), src.size(1), static_cast<int>(src.size(1)), w, wigner_floats,
        ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
        cols_per_channel, cols_per_channel - 1, blocks, dst.data_ptr<float>(), copy_set, radial_set,
        n_edges, static_cast<int>(n_channels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
  }
  channel_rotate_to_blocks_kernel<<<grid, kChannelThreads, 0, stream>>>(
      src.data_ptr<float>(), src.size(1), w,
      ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
      cols_per_channel, cols_per_channel - 1, blocks, dst.data_ptr<float>(), copy_set, radial_set,
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
    int64_t wigner_stride,
    int64_t copies,
    std::vector<torch::Tensor> radials,
    torch::Tensor plain,
    torch::Tensor radial_grad,
    std::vector<int64_t> radial_offsets,
    torch::Tensor dot_src,
    torch::Tensor dot_out,
    torch::Tensor sum_out) {
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
  const Copies copy_set{row_of_edge.numel() > 0 ? row_of_edge.data_ptr<int64_t>() : nullptr,
                        edge_scale.numel() > 0 ? edge_scale.data_ptr<float>() : nullptr,
                        src.numel() / copies, static_cast<int>(copies)};
  GatherExtras extras;
  extras.radial = make_radial_set(radials, plain.numel() > 0 ? plain.data_ptr<float>() : nullptr);
  extras.scale_radial = false;
  for (int m = 0; m < kMaxBlocks; ++m) {
    extras.scale_radial = extras.scale_radial || extras.radial.ptr[m] != nullptr;
    extras.radial_offset[m] = m < static_cast<int>(radial_offsets.size()) ? static_cast<int>(radial_offsets[m]) : 0;
  }
  extras.radial_grad = radial_grad.numel() > 0 ? radial_grad.data_ptr<float>() : nullptr;
  extras.radial_grad_stride = radial_grad.numel() > 0 ? radial_grad.stride(0) : 0;
  extras.dot_src = dot_src.numel() > 0 ? dot_src.data_ptr<float>() : nullptr;
  extras.dot_out = dot_out.numel() > 0 ? dot_out.data_ptr<float>() : nullptr;
  extras.sum_out = sum_out.numel() > 0 ? sum_out.data_ptr<float>() : nullptr;
  TORCH_CHECK(extras.sum_out == nullptr || extras.scale_radial, "sum_out is written with the radial scaling");
  const bool needs_edge_kernel = extras.scale_radial || extras.dot_src != nullptr;
  // Edge-tiled gather for identity or compact Wigner data that fit in shared memory.
  const int64_t total_width = n_edges > 0 ? src.numel() / copies / n_edges : 0;
  const int n_blocks = cols_per_channel;
  const int wigner_floats = w.mode == 2 ? static_cast<int>(w.compact_stride) : 0;
  const int64_t edge_bytes = (total_width + wigner_floats + dst.size(1)) * static_cast<int64_t>(sizeof(float));
  if ((w.mode == 0 || w.mode == 2) && n_edges <= INT32_MAX && n_channels <= INT32_MAX &&
      total_width * n_edges * copies == src.numel() &&
      allow_shared_memory(channel_gather_edge_kernel, edge_bytes)) {
    channel_gather_edge_kernel<<<static_cast<unsigned int>(n_edges), edge_block_threads(n_channels), edge_bytes,
                                 stream>>>(
        src.data_ptr<float>(), w, wigner_floats,
        ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
        cols_per_channel, cols_per_channel - 1, blocks, n_blocks, static_cast<int>(total_width),
        dst.data_ptr<float>(), dst.size(1), static_cast<int>(dst.size(1)), zero_fill, copy_set, extras,
        accumulate, n_edges, static_cast<int>(n_channels));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dst;
  }
  TORCH_CHECK(!needs_edge_kernel, "radial or dot gather extras need the edge-tiled kernel (compact Wigner data "
              "whose staged row fits in shared memory)");
  channel_gather_from_blocks_kernel<<<grid, kChannelThreads, 0, stream>>>(
      src.data_ptr<float>(), w,
      ch_base.data_ptr<int32_t>(), ch_l.data_ptr<int32_t>(), ch_cols.data_ptr<int32_t>(),
      cols_per_channel, cols_per_channel - 1, blocks, dst.data_ptr<float>(), dst.size(1), copy_set,
      accumulate, n_edges, n_channels);
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
