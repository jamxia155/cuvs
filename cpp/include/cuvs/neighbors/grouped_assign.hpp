/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

#include <vector>

namespace CUVS_EXPORT cuvs {
namespace neighbors {
namespace grouped_assign {

/**
 * @defgroup grouped_assign_cpp Grouped-GEMM nearest-centroid assignment
 * @{
 *
 * Approximate top-1 assignment of a large batch of vectors to a fixed set of `n_centroids`
 * centroids. The centroids are grouped into `n_lists` lists (about sqrt(n_centroids)) by k-means.
 * A batch is assigned by (1) scoring the list centers and taking each vector's `n_probes` best
 * lists, (2) bucketing the (list, vector) pairs by list, (3) for each list gathering its vectors
 * and running one dense TF32 GEMM against that list's centroids plus a row argmax, and (4)
 * merging each vector's per-list winners. The cost per vector is about
 * (n_lists + n_probes * n_centroids / n_lists) * dim multiply-adds on tensor cores, versus
 * n_centroids * dim for an exhaustive scan.
 */

struct index_params {
  /** Number of lists. 0 means round(sqrt(n_centroids)). */
  uint32_t n_lists = 0;
  /** Lists probed per vector. */
  uint32_t n_probes = 8;
  /** Lloyd iterations used to group the centroids. */
  uint32_t kmeans_n_iters = 10;
  /**
   * Each list's GEMM is rounded up to a multiple of this many centroids (the extra columns are
   * computed and ignored). 256 matches the tensor-core tile size and was fastest on H100.
   */
  uint32_t tile_align = 256;
  /**
   * L2Expanded / L2SqrtExpanded: nearest by L2. CosineExpanded: nearest by angle (the centroids
   * are normalized; vectors need not be). InnerProduct: largest inner product.
   */
  cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded;
};

struct index {
  index(raft::resources const& res);
  index(index&&)            = default;
  index& operator=(index&&) = default;

  uint32_t n_centroids = 0;
  uint32_t dim         = 0;
  uint32_t n_lists     = 0;
  uint32_t n_probes    = 0;
  uint32_t tile_align  = 0;
  /** True when scores are x.c - |c|^2/2 (L2); false when they are plain x.c. */
  bool l2 = false;

  /** [n_lists, dim] list centers. */
  raft::device_matrix<float, int64_t, raft::row_major> list_centers;
  /** [n_lists] |center|^2/2 (L2 only). */
  raft::device_vector<float, int64_t> list_bias;
  /** [n_centroids + tile_align, dim] centroids, each list contiguous, trailing rows zero. */
  raft::device_matrix<float, int64_t, raft::row_major> perm_centroids;
  /** [n_centroids + tile_align] |centroid|^2/2 in the same order (L2 only). */
  raft::device_vector<float, int64_t> perm_bias;
  /** [n_centroids] original centroid id of each row of perm_centroids. */
  raft::device_vector<int32_t, int64_t> perm_id;
  /** Host: first row of each list in perm_centroids, and its size. */
  std::vector<int32_t> list_offset;
  std::vector<int32_t> list_size;
};

/** Groups `centroids` ([n_centroids, dim], f32) into lists. */
auto build(raft::resources const& res,
           const index_params& params,
           raft::device_matrix_view<const float, int64_t, raft::row_major> centroids) -> index;

/**
 * Writes the id (row of the `centroids` passed to `build`) of each query's approximately nearest
 * centroid into `labels`. Enqueued on the stream of `res`; it synchronizes that stream once.
 */
void assign(raft::resources const& res,
            const index& idx,
            raft::device_matrix_view<const float, int64_t, raft::row_major> queries,
            raft::device_vector_view<uint32_t, int64_t> labels);

/** @} */

}  // namespace grouped_assign
}  // namespace neighbors
}  // namespace CUVS_EXPORT cuvs
