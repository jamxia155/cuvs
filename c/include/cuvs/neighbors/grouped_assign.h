/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/c_api.h>
#include <cuvs/core/export.h>
#include <cuvs/distance/distance.h>
#include <dlpack/dlpack.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup grouped_assign_c Grouped-GEMM nearest-centroid assignment
 * @{
 *
 * Approximate top-1 assignment of large batches of float32 vectors to a fixed set of centroids,
 * by grouping the centroids into ~sqrt(K) lists and running one dense TF32 GEMM per probed list.
 */

/** Opaque handle to a grouped-assign index. */
typedef struct {
  uintptr_t addr;
} cuvsGroupedAssignIndex;

typedef cuvsGroupedAssignIndex* cuvsGroupedAssignIndex_t;

CUVS_EXPORT cuvsError_t cuvsGroupedAssignIndexCreate(cuvsGroupedAssignIndex_t* index);

CUVS_EXPORT cuvsError_t cuvsGroupedAssignIndexDestroy(cuvsGroupedAssignIndex_t index);

/**
 * @brief Group `centroids` into lists.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] centroids DLManagedTensor* float32 [n_centroids, dim] on device; dim % 4 == 0
 * @param[in] metric L2Expanded, L2SqrtExpanded, CosineExpanded or InnerProduct
 * @param[in] n_lists number of lists; 0 means round(sqrt(n_centroids))
 * @param[in] n_probes lists probed per vector
 * @param[in] kmeans_n_iters Lloyd iterations used to group the centroids
 * @param[in] tile_align each list's GEMM is rounded up to a multiple of this many centroids (256)
 * @param[out] index built index
 */
CUVS_EXPORT cuvsError_t cuvsGroupedAssignBuild(cuvsResources_t res,
                                               DLManagedTensor* centroids,
                                               cuvsDistanceType metric,
                                               uint32_t n_lists,
                                               uint32_t n_probes,
                                               uint32_t kmeans_n_iters,
                                               uint32_t tile_align,
                                               cuvsGroupedAssignIndex_t index);

/**
 * @brief Write each query's (approximately) nearest centroid id into `labels`.
 *
 * Enqueued on the stream of `res`, which it synchronizes once.
 *
 * @param[in] res cuvsResources_t opaque C handle
 * @param[in] index built index
 * @param[in] queries DLManagedTensor* float32 [n_queries, dim] on device
 * @param[out] labels DLManagedTensor* uint32 [n_queries] (or [n_queries, 1]) on device
 */
CUVS_EXPORT cuvsError_t cuvsGroupedAssign(cuvsResources_t res,
                                          cuvsGroupedAssignIndex_t index,
                                          DLManagedTensor* queries,
                                          DLManagedTensor* labels);

/** @} */

#ifdef __cplusplus
}
#endif
