/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/grouped_assign.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/cuda_utils.cuh>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace cuvs::neighbors::grouped_assign {

namespace {

using KV = cub::KeyValuePair<int, float>;

struct ArgMaxOp {
  __device__ KV operator()(const KV& a, const KV& b) const
  {
    return (b.value > a.value || (b.value == a.value && b.key < a.key)) ? b : a;
  }
};

constexpr int kBlock = 256;

// score(r, c) = scores[r * stride + c] - (bias ? bias[c] : 0). One block per row.
__global__ void argmax_rows_kernel(
  const float* scores, int cols, int stride, const float* bias, int* arg)
{
  const float* row = scores + (size_t)blockIdx.x * stride;
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = row[c] - (bias ? bias[c] : 0.f);
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, kBlock>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) arg[blockIdx.x] = r.key;
}

// Like argmax_rows_kernel, but writes the (id_map'd) winner of pair `pairs[row]`.
__global__ void argmax_scatter_kernel(const float* scores,
                                      int cols,
                                      int stride,
                                      const float* bias,
                                      const int* pairs,
                                      const int* id_map,
                                      float* res_score,
                                      int* res_id)
{
  const float* row = scores + (size_t)blockIdx.x * stride;
  KV loc(0, -INFINITY);
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = row[c] - (bias ? bias[c] : 0.f);
    if (v > loc.value) loc = KV(c, v);
  }
  using BR = cub::BlockReduce<KV, kBlock>;
  __shared__ typename BR::TempStorage t;
  KV r = BR(t).Reduce(loc, ArgMaxOp());
  if (threadIdx.x == 0) {
    int slot        = pairs[blockIdx.x];
    res_score[slot] = r.value;
    res_id[slot]    = id_map[r.key];
  }
}

// Each row's `p` best columns (after the bias), best first.
__global__ void top_p_kernel(
  const float* scores, int cols, int stride, const float* bias, int p, int* out)
{
  extern __shared__ float row[];
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    row[c] = scores[(size_t)blockIdx.x * stride + c] - (bias ? bias[c] : 0.f);
  __syncthreads();
  using BR = cub::BlockReduce<KV, kBlock>;
  __shared__ typename BR::TempStorage t;
  for (int j = 0; j < p; j++) {
    KV loc(0, -INFINITY);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
      if (row[c] > loc.value) loc = KV(c, row[c]);
    KV r = BR(t).Reduce(loc, ArgMaxOp());
    if (threadIdx.x == 0) {
      out[(size_t)blockIdx.x * p + j] = r.key;
      row[r.key]                      = -INFINITY;
    }
    __syncthreads();
  }
}

__global__ void init_pairs_kernel(const int* lists, int n, int* keys, int* vals)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    keys[i] = lists[i];
    vals[i] = i;
  }
}

__global__ void count_keys_kernel(const int* keys, int n, int* counts)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(counts + keys[i], 1);
}

// Pair `sorted_vals[off + r]` is (query * p + probe); copy that query's row into out[r].
__global__ void gather_queries_kernel(
  const float* q, int dim, const int* sorted_vals, int off, int p, float* out)
{
  int query         = sorted_vals[off + blockIdx.x] / p;
  const float4* src = reinterpret_cast<const float4*>(q + (size_t)query * dim);
  float4* dst       = reinterpret_cast<float4*>(out + (size_t)blockIdx.x * dim);
  for (int c = threadIdx.x; c < dim / 4; c += blockDim.x)
    dst[c] = src[c];
}

__global__ void merge_pairs_kernel(
  const float* res_score, const int* res_id, int p, int nq, uint32_t* labels)
{
  int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= nq) return;
  float best = -INFINITY;
  int arg    = 0;
  for (int j = 0; j < p; j++) {
    float s = res_score[(size_t)q * p + j];
    if (s > best) {
      best = s;
      arg  = res_id[(size_t)q * p + j];
    }
  }
  labels[q] = static_cast<uint32_t>(arg);
}

__global__ void accumulate_kernel(
  const float* x, const int* assign, int dim, float* sums, int* counts)
{
  int a = assign[blockIdx.x];
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    atomicAdd(sums + (size_t)a * dim + c, x[(size_t)blockIdx.x * dim + c]);
  if (threadIdx.x == 0) atomicAdd(counts + a, 1);
}

// Row r of x scaled to unit norm (a zero row is left alone).
__global__ void normalize_rows_kernel(float* x, int dim)
{
  float* row = x + (size_t)blockIdx.x * dim;
  float s    = 0;
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    s += row[c] * row[c];
  using BR = cub::BlockReduce<float, kBlock>;
  __shared__ typename BR::TempStorage t;
  __shared__ float inv;
  float tot = BR(t).Sum(s);
  if (threadIdx.x == 0) inv = tot > 0 ? rsqrtf(tot) : 1.0f;
  __syncthreads();
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    row[c] *= inv;
}

// sums[m] / counts[m] (L2 means); an empty list keeps its old center.
__global__ void mean_rows_kernel(const float* sums, const int* counts, int dim, float* centers)
{
  int n = counts[blockIdx.x];
  if (n == 0) return;
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    centers[(size_t)blockIdx.x * dim + c] = sums[(size_t)blockIdx.x * dim + c] / n;
}

__global__ void half_sq_norm_kernel(const float* x, int dim, float* out)
{
  const float* row = x + (size_t)blockIdx.x * dim;
  float s          = 0;
  for (int c = threadIdx.x; c < dim; c += blockDim.x)
    s += row[c] * row[c];
  using BR = cub::BlockReduce<float, kBlock>;
  __shared__ typename BR::TempStorage t;
  float tot = BR(t).Sum(s);
  if (threadIdx.x == 0) out[blockIdx.x] = 0.5f * tot;
}

// C (column-major [n, m], leading dimension ldc) = A^T B for row-major A [n, dim] and
// B [m, dim]; i.e. row-major C[m, n] = B A^T. TF32 tensor cores, fp32 accumulate.
void gemm_tf32(raft::resources const& res,
               int n,
               int m,
               int dim,
               const float* A,
               const float* B,
               float* C,
               int ldc)
{
  const float one = 1.f, zero = 0.f;
  cublasHandle_t h = raft::resource::get_cublas_handle(res);
  RAFT_CUBLAS_TRY(cublasSetStream(h, raft::resource::get_cuda_stream(res)));
  RAFT_CUBLAS_TRY(cublasGemmEx(h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               n,
                               m,
                               dim,
                               &one,
                               A,
                               CUDA_R_32F,
                               dim,
                               B,
                               CUDA_R_32F,
                               dim,
                               &zero,
                               C,
                               CUDA_R_32F,
                               ldc,
                               CUBLAS_COMPUTE_32F_FAST_TF32,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

inline int round_up(int x, int a) { return (x + a - 1) / a * a; }

}  // namespace

index::index(raft::resources const& res)
  : list_centers(raft::make_device_matrix<float, int64_t>(res, 0, 0)),
    list_bias(raft::make_device_vector<float, int64_t>(res, 0)),
    perm_centroids(raft::make_device_matrix<float, int64_t>(res, 0, 0)),
    perm_bias(raft::make_device_vector<float, int64_t>(res, 0)),
    perm_id(raft::make_device_vector<int32_t, int64_t>(res, 0))
{
}

auto build(raft::resources const& res,
           const index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> centroids) -> index
{
  using cuvs::distance::DistanceType;
  const int K   = static_cast<int>(centroids.extent(0));
  const int dim = static_cast<int>(centroids.extent(1));
  RAFT_EXPECTS(K > 0 && dim > 0 && dim % 4 == 0, "grouped_assign needs dim divisible by 4");
  RAFT_EXPECTS(params.tile_align >= 1, "tile_align must be positive");
  const bool l2 = params.metric == DistanceType::L2Expanded ||
                  params.metric == DistanceType::L2SqrtExpanded;
  const bool cosine = params.metric == DistanceType::CosineExpanded;
  RAFT_EXPECTS(l2 || cosine || params.metric == DistanceType::InnerProduct,
               "grouped_assign supports L2, cosine and inner-product metrics");
  const int M = std::clamp<int>(
    params.n_lists ? static_cast<int>(params.n_lists) : static_cast<int>(std::lround(std::sqrt((double)K))),
    1,
    K);
  const int align = static_cast<int>(params.tile_align);
  auto stream     = raft::resource::get_cuda_stream(res);

  index idx(res);
  idx.n_centroids = K;
  idx.dim         = dim;
  idx.n_lists     = M;
  idx.n_probes    = std::clamp<int>(static_cast<int>(params.n_probes), 1, M);
  idx.tile_align  = align;
  idx.l2          = l2;

  // Working copy of the centroids (normalized for cosine).
  auto cents = raft::make_device_matrix<float, int64_t>(res, K, dim);
  raft::copy(cents.data_handle(), centroids.data_handle(), (size_t)K * dim, stream);
  if (cosine) normalize_rows_kernel<<<K, kBlock, 0, stream>>>(cents.data_handle(), dim);

  // ---- Group the centroids into M lists with Lloyd iterations. ----
  auto centers = raft::make_device_matrix<float, int64_t>(res, M, dim);
  auto cbias   = raft::make_device_vector<float, int64_t>(res, M);
  auto sums    = raft::make_device_matrix<float, int64_t>(res, M, dim);
  auto scores  = raft::make_device_matrix<float, int64_t>(res, K, M);
  auto assign  = raft::make_device_vector<int, int64_t>(res, K);
  auto counts  = raft::make_device_vector<int, int64_t>(res, M);
  {
    std::mt19937 rng(7);
    std::vector<int> pick(K);
    std::iota(pick.begin(), pick.end(), 0);
    std::shuffle(pick.begin(), pick.end(), rng);
    for (int m = 0; m < M; m++)
      raft::copy(centers.data_handle() + (size_t)m * dim,
                 cents.data_handle() + (size_t)pick[m] * dim,
                 dim,
                 stream);
  }
  auto assign_all = [&] {
    if (l2) half_sq_norm_kernel<<<M, kBlock, 0, stream>>>(centers.data_handle(), dim, cbias.data_handle());
    gemm_tf32(res, M, K, dim, centers.data_handle(), cents.data_handle(), scores.data_handle(), M);
    argmax_rows_kernel<<<K, kBlock, 0, stream>>>(
      scores.data_handle(), M, M, l2 ? cbias.data_handle() : nullptr, assign.data_handle());
  };
  for (uint32_t it = 0; it < params.kmeans_n_iters; it++) {
    assign_all();
    RAFT_CUDA_TRY(cudaMemsetAsync(sums.data_handle(), 0, (size_t)M * dim * 4, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(counts.data_handle(), 0, (size_t)M * 4, stream));
    accumulate_kernel<<<K, kBlock, 0, stream>>>(
      cents.data_handle(), assign.data_handle(), dim, sums.data_handle(), counts.data_handle());
    if (l2) {
      mean_rows_kernel<<<M, kBlock, 0, stream>>>(
        sums.data_handle(), counts.data_handle(), dim, centers.data_handle());
    } else {
      // Spherical: normalize the sums; an empty list (zero sum) keeps its old center.
      auto old = raft::make_device_matrix<float, int64_t>(res, M, dim);
      raft::copy(old.data_handle(), centers.data_handle(), (size_t)M * dim, stream);
      normalize_rows_kernel<<<M, kBlock, 0, stream>>>(sums.data_handle(), dim);
      std::vector<int> hc(M);
      raft::update_host(hc.data(), counts.data_handle(), M, stream);
      raft::resource::sync_stream(res);
      for (int m = 0; m < M; m++)
        if (hc[m] > 0)
          raft::copy(centers.data_handle() + (size_t)m * dim,
                     sums.data_handle() + (size_t)m * dim,
                     dim,
                     stream);
    }
  }
  assign_all();  // final lists (and list biases) match the final centers

  std::vector<int> h_assign(K);
  raft::update_host(h_assign.data(), assign.data_handle(), K, stream);
  raft::resource::sync_stream(res);

  // ---- Lists in CSR form, centroids permuted so each list is contiguous. ----
  idx.list_size.assign(M, 0);
  idx.list_offset.assign(M + 1, 0);
  for (int a : h_assign)
    idx.list_size[a]++;
  for (int m = 0; m < M; m++)
    idx.list_offset[m + 1] = idx.list_offset[m] + idx.list_size[m];
  std::vector<int> fill(idx.list_offset.begin(), idx.list_offset.end() - 1), h_perm(K);
  for (int c = 0; c < K; c++) {
    int pos     = fill[h_assign[c]]++;
    h_perm[pos] = c;
  }

  idx.list_centers = raft::make_device_matrix<float, int64_t>(res, M, dim);
  raft::copy(idx.list_centers.data_handle(), centers.data_handle(), (size_t)M * dim, stream);
  idx.list_bias = raft::make_device_vector<float, int64_t>(res, l2 ? M : 0);
  if (l2) {
    half_sq_norm_kernel<<<M, kBlock, 0, stream>>>(centers.data_handle(), dim, idx.list_bias.data_handle());
  }
  idx.perm_id = raft::make_device_vector<int32_t, int64_t>(res, K);
  raft::update_device(idx.perm_id.data_handle(), h_perm.data(), K, stream);
  // Slack rows: a list's GEMM may run up to `align` columns past its end.
  const int64_t rows = (int64_t)K + align;
  idx.perm_centroids = raft::make_device_matrix<float, int64_t>(res, rows, dim);
  RAFT_CUDA_TRY(cudaMemsetAsync(idx.perm_centroids.data_handle(), 0, (size_t)rows * dim * 4, stream));
  // Rows of `cents` in list order (cosine: normalized). Row gather via per-row copies.
  for (int p = 0; p < K; p++)
    raft::copy(idx.perm_centroids.data_handle() + (size_t)p * dim,
               cents.data_handle() + (size_t)h_perm[p] * dim,
               dim,
               stream);
  idx.perm_bias = raft::make_device_vector<float, int64_t>(res, l2 ? rows : 0);
  if (l2) {
    half_sq_norm_kernel<<<rows, kBlock, 0, stream>>>(
      idx.perm_centroids.data_handle(), dim, idx.perm_bias.data_handle());
  }
  raft::resource::sync_stream(res);
  return idx;
}

void assign(raft::resources const& res,
            const index& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_vector_view<uint32_t, int64_t> labels)
{
  const int B   = static_cast<int>(queries.extent(0));
  const int dim = static_cast<int>(idx.dim);
  RAFT_EXPECTS(queries.extent(1) == dim, "query dimension does not match the index");
  RAFT_EXPECTS(labels.extent(0) >= B, "labels is too small");
  if (B == 0) return;
  const int M     = static_cast<int>(idx.n_lists);
  const int P     = static_cast<int>(idx.n_probes);
  const int align = static_cast<int>(idx.tile_align);
  const int64_t NP = (int64_t)B * P;
  RAFT_EXPECTS(NP < (int64_t{1} << 31), "batch * n_probes is too large");
  auto stream = raft::resource::get_cuda_stream(res);

  // (1) list scores and each query's top-P lists
  const int Mpad = round_up(M, 8);
  auto s1        = raft::make_device_matrix<float, int64_t>(res, B, Mpad);
  auto probes    = raft::make_device_vector<int, int64_t>(res, NP);
  gemm_tf32(res, M, B, dim, idx.list_centers.data_handle(), queries.data_handle(), s1.data_handle(), Mpad);
  top_p_kernel<<<B, kBlock, (size_t)M * 4, stream>>>(
    s1.data_handle(), M, Mpad, idx.l2 ? idx.list_bias.data_handle() : nullptr, P, probes.data_handle());

  // (2) bucket the (list, query) pairs by list
  auto keys        = raft::make_device_vector<int, int64_t>(res, NP);
  auto vals        = raft::make_device_vector<int, int64_t>(res, NP);
  auto keys_sorted = raft::make_device_vector<int, int64_t>(res, NP);
  auto vals_sorted = raft::make_device_vector<int, int64_t>(res, NP);
  auto d_counts    = raft::make_device_vector<int, int64_t>(res, M);
  init_pairs_kernel<<<(NP + 255) / 256, 256, 0, stream>>>(
    probes.data_handle(), (int)NP, keys.data_handle(), vals.data_handle());
  RAFT_CUDA_TRY(cudaMemsetAsync(d_counts.data_handle(), 0, (size_t)M * 4, stream));
  count_keys_kernel<<<(NP + 255) / 256, 256, 0, stream>>>(
    keys.data_handle(), (int)NP, d_counts.data_handle());
  const int end_bit = std::max(1, (int)std::ceil(std::log2((double)M)));
  size_t sort_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(nullptr,
                                                sort_bytes,
                                                keys.data_handle(),
                                                keys_sorted.data_handle(),
                                                vals.data_handle(),
                                                vals_sorted.data_handle(),
                                                (int)NP,
                                                0,
                                                end_bit,
                                                stream));
  auto sort_tmp = raft::make_device_vector<uint8_t, int64_t>(res, sort_bytes);
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(sort_tmp.data_handle(),
                                                sort_bytes,
                                                keys.data_handle(),
                                                keys_sorted.data_handle(),
                                                vals.data_handle(),
                                                vals_sorted.data_handle(),
                                                (int)NP,
                                                0,
                                                end_bit,
                                                stream));
  std::vector<int> counts(M);
  raft::update_host(counts.data(), d_counts.data_handle(), M, stream);
  raft::resource::sync_stream(res);

  // (3) per list: gather its queries, one GEMM against its centroids, row argmax
  int max_g = 0, max_cols = 0;
  for (int m = 0; m < M; m++) {
    if (counts[m] > 0 && idx.list_size[m] > 0) {
      max_g    = std::max(max_g, counts[m]);
      max_cols = std::max(max_cols, round_up(idx.list_size[m], align));
    }
  }
  auto res_score = raft::make_device_vector<float, int64_t>(res, NP);
  auto res_id    = raft::make_device_vector<int, int64_t>(res, NP);
  if (max_g > 0) {
    auto qg = raft::make_device_matrix<float, int64_t>(res, max_g, dim);
    auto sc = raft::make_device_matrix<float, int64_t>(res, max_g, max_cols);
    int off = 0;
    for (int m = 0; m < M; m++) {
      const int g = counts[m], n = idx.list_size[m];
      if (g > 0 && n > 0) {
        const int npad = round_up(n, align);
        const int row0 = idx.list_offset[m];
        gather_queries_kernel<<<g, kBlock, 0, stream>>>(
          queries.data_handle(), dim, vals_sorted.data_handle(), off, P, qg.data_handle());
        gemm_tf32(res,
                  npad,
                  g,
                  dim,
                  idx.perm_centroids.data_handle() + (size_t)row0 * dim,
                  qg.data_handle(),
                  sc.data_handle(),
                  npad);
        argmax_scatter_kernel<<<g, kBlock, 0, stream>>>(
          sc.data_handle(),
          n,
          npad,
          idx.l2 ? idx.perm_bias.data_handle() + row0 : nullptr,
          vals_sorted.data_handle() + off,
          idx.perm_id.data_handle() + row0,
          res_score.data_handle(),
          res_id.data_handle());
      }
      off += g;
    }
  }

  // (4) merge each query's P partial winners
  merge_pairs_kernel<<<(B + 255) / 256, 256, 0, stream>>>(
    res_score.data_handle(), res_id.data_handle(), P, B, labels.data_handle());
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::grouped_assign
