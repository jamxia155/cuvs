/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../ivf_pq_encode_codes.hpp"

#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace cuvs::neighbors::ivf_pq::detail {

namespace {

constexpr uint32_t kBook   = 256;  // pq_bits == 8
constexpr uint32_t kJChunk = 16;   // subspaces handled by one block (output is staged per chunk)

// ---------------------------------------------------------------------------------------------
// SIMT path.
//
// A block takes 64 rows and a chunk of 16 subspaces. For each subspace the codebook slice
// [pq_len, 256] is staged in shared memory; warp w owns rows 8w..8w+7 and lane l owns codes
// {4l..4l+3, 128+4l..128+4l+3}, so a thread keeps an 8x8 tile of distances in registers and every
// shared-memory load feeds several FMAs. The per-code distance is the same `fma(t, t, d)` chain over
// k as the baseline kernel. The argmin breaks exact ties towards the smaller code.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t kSimtRows    = 64;
constexpr uint32_t kSimtThreads = 256;

template <uint32_t PqLenT>
__global__ void __launch_bounds__(kSimtThreads) encode_simt_kernel(const float* __restrict__ residuals,
                                                                   int64_t n_rows,
                                                                   uint32_t pq_dim,
                                                                   uint32_t pq_len_rt,
                                                                   const float* __restrict__ pq_centers,
                                                                   uint8_t* __restrict__ out,
                                                                   uint32_t out_stride)
{
  const uint32_t pq_len = PqLenT != 0 ? PqLenT : pq_len_rt;
  extern __shared__ __align__(16) float smem[];
  float* sC     = smem;                       // [pq_len, 256]
  float* sX     = sC + pq_len * kBook;        // [64, pq_len]
  auto* sOut    = reinterpret_cast<uint8_t*>(sX + kSimtRows * pq_len);  // [64, kJChunk]

  const int64_t row0 = int64_t{blockIdx.x} * kSimtRows;
  const uint32_t j0  = blockIdx.y * kJChunk;
  const uint32_t jn  = min(kJChunk, pq_dim - j0);
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const size_t rot_dim = size_t{pq_dim} * pq_len;

  for (uint32_t jj = 0; jj < jn; jj++) {
    const uint32_t j = j0 + jj;
    __syncthreads();  // the previous subspace is done with the shared tiles
    {
      const auto* src = reinterpret_cast<const float4*>(pq_centers + size_t{j} * pq_len * kBook);
      auto* dst       = reinterpret_cast<float4*>(sC);
      for (uint32_t i = threadIdx.x; i < pq_len * kBook / 4; i += kSimtThreads) {
        dst[i] = src[i];
      }
      for (uint32_t i = threadIdx.x; i < kSimtRows * pq_len; i += kSimtThreads) {
        const int64_t row = row0 + i / pq_len;
        sX[i] = row < n_rows
                  ? residuals[static_cast<size_t>(row) * rot_dim + size_t{j} * pq_len + i % pq_len]
                  : 0.0f;
      }
    }
    __syncthreads();

    float acc[8][8];
#pragma unroll
    for (int r = 0; r < 8; r++) {
#pragma unroll
      for (int c = 0; c < 8; c++) {
        acc[r][c] = 0.0f;
      }
    }
#pragma unroll
    for (uint32_t k = 0; k < pq_len; k++) {
      const float4 c0 = *reinterpret_cast<const float4*>(sC + k * kBook + lane * 4);
      const float4 c1 = *reinterpret_cast<const float4*>(sC + k * kBook + 128 + lane * 4);
      const float cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
#pragma unroll
      for (int r = 0; r < 8; r++) {
        const float x = sX[(warp * 8 + r) * pq_len + k];
#pragma unroll
        for (int c = 0; c < 8; c++) {
          const float t = x - cv[c];
          acc[r][c]     = fmaf(t, t, acc[r][c]);
        }
      }
    }

#pragma unroll
    for (int r = 0; r < 8; r++) {
      float bd    = acc[r][0];
      uint32_t bi = lane * 4;
#pragma unroll
      for (int c = 1; c < 8; c++) {
        const uint32_t idx = c < 4 ? lane * 4 + c : 128 + lane * 4 + (c - 4);
        if (acc[r][c] < bd) {
          bd = acc[r][c];
          bi = idx;
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) {
        const float od    = __shfl_xor_sync(0xffffffffu, bd, off);
        const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (od < bd || (od == bd && oi < bi)) {
          bd = od;
          bi = oi;
        }
      }
      if (lane == 0) { sOut[(warp * 8 + r) * kJChunk + jj] = static_cast<uint8_t>(bi); }
    }
  }
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < kSimtRows * jn; i += kSimtThreads) {
    const uint32_t r  = i / jn;
    const uint32_t jj = i % jn;
    const int64_t row = row0 + r;
    if (row < n_rows) {
      out[static_cast<size_t>(row) * out_stride + j0 + jj] = sOut[r * kJChunk + jj];
    }
  }
}

template <uint32_t PqLenT>
void launch_simt(cudaStream_t stream,
                 const float* residuals,
                 int64_t n_rows,
                 uint32_t pq_dim,
                 uint32_t pq_len,
                 const float* pq_centers,
                 uint8_t* out,
                 uint32_t out_stride)
{
  const size_t smem =
    (size_t{pq_len} * kBook + size_t{kSimtRows} * pq_len) * sizeof(float) + kSimtRows * kJChunk;
  auto* kernel = encode_simt_kernel<PqLenT>;
  RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const dim3 grid(static_cast<uint32_t>((n_rows + kSimtRows - 1) / kSimtRows),
                  (pq_dim + kJChunk - 1) / kJChunk);
  kernel<<<grid, kSimtThreads, smem, stream>>>(
    residuals, n_rows, pq_dim, pq_len, pq_centers, out, out_stride);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// ---------------------------------------------------------------------------------------------
// Tensor-core path.
//
// argmin_l |x - c_l|^2 = argmin_l (|c_l|^2 - 2 x.c_l). For a subspace the dot products of 16 rows
// with 8 codes are one mma.m16n8k8 (TF32) per 8 columns of pq_len. A warp owns 16 rows, a block
// 128 rows and a chunk of 16 subspaces. The warp walks the 32 tiles of 8 codes, folds each tile's
// scores into a running (min, code) per row, and finally reduces over the 4 lanes of a quad that
// share a row. Inputs are rounded to TF32 (cvt.rna); with `Passes == 3` each operand is split into
// hi + lo TF32 parts and hi*hi + hi*lo + lo*hi is accumulated, which is close to FP32 accuracy.
// ---------------------------------------------------------------------------------------------
constexpr uint32_t kTcRows    = 128;
constexpr uint32_t kTcThreads = 256;
constexpr uint32_t kTcStride  = kBook + 8;  // padded row of the staged codebook: no bank conflicts

__device__ __forceinline__ uint32_t to_tf32(float x)
{
  uint32_t r;
  asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
  return r;
}

__device__ __forceinline__ void mma_tf32(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
    "{%0,%1,%2,%3};"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
  __trap();
#endif
}

template <uint32_t KSteps, int Passes>
__global__ void __launch_bounds__(kTcThreads) encode_tensor_kernel(const float* __restrict__ residuals,
                                                                   int64_t n_rows,
                                                                   uint32_t pq_dim,
                                                                   const float* __restrict__ pq_centers,
                                                                   uint8_t* __restrict__ out,
                                                                   uint32_t out_stride)
{
  constexpr uint32_t pq_len = KSteps * 8;
  extern __shared__ __align__(16) float smem[];
  float* sCh   = smem;                                              // [pq_len, kTcStride]
  [[maybe_unused]] float* sCl = sCh + (Passes == 3 ? pq_len * kTcStride : 0);  // lo parts (3 passes)
  float* sBias = sCl + pq_len * kTcStride;                          // [256], follows sCh or sCl
  auto* sOut   = reinterpret_cast<uint8_t*>(sBias + kBook);         // [128, kJChunk]

  const int64_t row0  = int64_t{blockIdx.x} * kTcRows;
  const uint32_t j0   = blockIdx.y * kJChunk;
  const uint32_t jn   = min(kJChunk, pq_dim - j0);
  const uint32_t warp = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t g    = lane / 4;  // mma "groupID": row within the 8-row half, B column
  const uint32_t q    = lane % 4;  // mma "threadID_in_group"
  const size_t rot_dim = size_t{pq_dim} * pq_len;

  for (uint32_t jj = 0; jj < jn; jj++) {
    const uint32_t j = j0 + jj;
    __syncthreads();
    const float* cb = pq_centers + size_t{j} * pq_len * kBook;
    for (uint32_t i = threadIdx.x; i < pq_len * kBook; i += kTcThreads) {
      const uint32_t k = i / kBook;
      const uint32_t l = i % kBook;
      const float c    = cb[i];
      const uint32_t h = to_tf32(c);
      sCh[k * kTcStride + l] = __uint_as_float(h);
      if constexpr (Passes == 3) {
        sCl[k * kTcStride + l] = __uint_as_float(to_tf32(c - __uint_as_float(h)));
      }
    }
    if (threadIdx.x < kBook) {
      float b = 0.0f;
      for (uint32_t k = 0; k < pq_len; k++) {
        const float c = cb[k * kBook + threadIdx.x];
        b             = fmaf(c, c, b);
      }
      sBias[threadIdx.x] = b;
    }

    // A fragments (16 rows x pq_len) of this warp, straight from global memory.
    uint32_t ah[KSteps][4];
    [[maybe_unused]] uint32_t al[KSteps][4];
    {
      const int64_t ra[2] = {row0 + warp * 16 + g, row0 + warp * 16 + g + 8};
#pragma unroll
      for (uint32_t kk = 0; kk < KSteps; kk++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int64_t row  = ra[e & 1];
          const uint32_t col = kk * 8 + q + ((e & 2) ? 4 : 0);
          const float x =
            row < n_rows
              ? residuals[static_cast<size_t>(row) * rot_dim + size_t{j} * pq_len + col]
              : 0.0f;
          ah[kk][e] = to_tf32(x);
          if constexpr (Passes == 3) { al[kk][e] = to_tf32(x - __uint_as_float(ah[kk][e])); }
        }
      }
    }
    __syncthreads();

    float best0 = std::numeric_limits<float>::infinity();
    float best1 = best0;
    uint32_t bi0 = 0;
    uint32_t bi1 = 0;
    for (uint32_t t = 0; t < kBook / 8; t++) {
      const uint32_t n0 = t * 8;
      float d[4]        = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (uint32_t kk = 0; kk < KSteps; kk++) {
        const uint32_t bh[2] = {__float_as_uint(sCh[(kk * 8 + q) * kTcStride + n0 + g]),
                                __float_as_uint(sCh[(kk * 8 + q + 4) * kTcStride + n0 + g])};
        mma_tf32(d, ah[kk], bh);
        if constexpr (Passes == 3) {
          const uint32_t bl[2] = {__float_as_uint(sCl[(kk * 8 + q) * kTcStride + n0 + g]),
                                  __float_as_uint(sCl[(kk * 8 + q + 4) * kTcStride + n0 + g])};
          mma_tf32(d, ah[kk], bl);
          mma_tf32(d, al[kk], bh);
        }
      }
      const float2 bias = *reinterpret_cast<const float2*>(sBias + n0 + 2 * q);
      const uint32_t c0 = n0 + 2 * q;
      const float s00   = fmaf(-2.0f, d[0], bias.x);
      const float s01   = fmaf(-2.0f, d[1], bias.y);
      const float s10   = fmaf(-2.0f, d[2], bias.x);
      const float s11   = fmaf(-2.0f, d[3], bias.y);
      if (s00 < best0) { best0 = s00; bi0 = c0; }
      if (s01 < best0) { best0 = s01; bi0 = c0 + 1; }
      if (s10 < best1) { best1 = s10; bi1 = c0; }
      if (s11 < best1) { best1 = s11; bi1 = c0 + 1; }
    }
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      const float ob0    = __shfl_xor_sync(0xffffffffu, best0, off);
      const uint32_t oi0 = __shfl_xor_sync(0xffffffffu, bi0, off);
      const float ob1    = __shfl_xor_sync(0xffffffffu, best1, off);
      const uint32_t oi1 = __shfl_xor_sync(0xffffffffu, bi1, off);
      if (ob0 < best0 || (ob0 == best0 && oi0 < bi0)) { best0 = ob0; bi0 = oi0; }
      if (ob1 < best1 || (ob1 == best1 && oi1 < bi1)) { best1 = ob1; bi1 = oi1; }
    }
    if (q == 0) {
      sOut[(warp * 16 + g) * kJChunk + jj]     = static_cast<uint8_t>(bi0);
      sOut[(warp * 16 + g + 8) * kJChunk + jj] = static_cast<uint8_t>(bi1);
    }
  }
  __syncthreads();
  for (uint32_t i = threadIdx.x; i < kTcRows * jn; i += kTcThreads) {
    const uint32_t r  = i / jn;
    const uint32_t jj = i % jn;
    const int64_t row = row0 + r;
    if (row < n_rows) {
      out[static_cast<size_t>(row) * out_stride + j0 + jj] = sOut[r * kJChunk + jj];
    }
  }
}

template <uint32_t KSteps, int Passes>
void launch_tensor(cudaStream_t stream,
                   const float* residuals,
                   int64_t n_rows,
                   uint32_t pq_dim,
                   const float* pq_centers,
                   uint8_t* out,
                   uint32_t out_stride)
{
  constexpr uint32_t pq_len = KSteps * 8;
  const size_t smem         = (size_t{Passes == 3 ? 2u : 1u} * pq_len * kTcStride + kBook) * sizeof(float) +
                      kTcRows * kJChunk;
  auto* kernel = encode_tensor_kernel<KSteps, Passes>;
  RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  const dim3 grid(static_cast<uint32_t>((n_rows + kTcRows - 1) / kTcRows),
                  (pq_dim + kJChunk - 1) / kJChunk);
  kernel<<<grid, kTcThreads, smem, stream>>>(residuals, n_rows, pq_dim, pq_centers, out, out_stride);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace

void ivf_pq_encode_simt(cudaStream_t stream,
                        const float* residuals,
                        int64_t n_rows,
                        uint32_t pq_dim,
                        uint32_t pq_len,
                        const float* pq_centers,
                        uint8_t* out,
                        uint32_t out_stride)
{
  RAFT_EXPECTS(pq_len >= 1 && pq_len <= 32,
               "the simt PQ encode needs 1 <= pq_len <= 32 (got %u)",
               pq_len);
  if (n_rows == 0) { return; }
  switch (pq_len) {
    case 4: launch_simt<4>(stream, residuals, n_rows, pq_dim, pq_len, pq_centers, out, out_stride); break;
    case 8: launch_simt<8>(stream, residuals, n_rows, pq_dim, pq_len, pq_centers, out, out_stride); break;
    case 16: launch_simt<16>(stream, residuals, n_rows, pq_dim, pq_len, pq_centers, out, out_stride); break;
    default: launch_simt<0>(stream, residuals, n_rows, pq_dim, pq_len, pq_centers, out, out_stride);
  }
}

void ivf_pq_encode_tensor(cudaStream_t stream,
                          const float* residuals,
                          int64_t n_rows,
                          uint32_t pq_dim,
                          uint32_t pq_len,
                          const float* pq_centers,
                          uint8_t* out,
                          uint32_t out_stride,
                          int tf32_passes)
{
  RAFT_EXPECTS(pq_len % 8 == 0 && pq_len >= 8 && pq_len <= 32,
               "the tensor PQ encode needs pq_len in {8, 16, 24, 32} (got %u)",
               pq_len);
  RAFT_EXPECTS(tf32_passes == 1 || tf32_passes == 3, "tf32_passes must be 1 or 3");
  if (n_rows == 0) { return; }
  const uint32_t ks = pq_len / 8;
#define CUVS_PQ_TC_LAUNCH(K)                                                                    \
  case K:                                                                                       \
    if (tf32_passes == 3) {                                                                     \
      launch_tensor<K, 3>(stream, residuals, n_rows, pq_dim, pq_centers, out, out_stride);      \
    } else {                                                                                    \
      launch_tensor<K, 1>(stream, residuals, n_rows, pq_dim, pq_centers, out, out_stride);      \
    }                                                                                           \
    break;
  switch (ks) {
    CUVS_PQ_TC_LAUNCH(1)
    CUVS_PQ_TC_LAUNCH(2)
    CUVS_PQ_TC_LAUNCH(3)
    CUVS_PQ_TC_LAUNCH(4)
    default: RAFT_FAIL("unreachable");
  }
#undef CUVS_PQ_TC_LAUNCH
}

}  // namespace cuvs::neighbors::ivf_pq::detail
