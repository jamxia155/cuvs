/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <dlpack/dlpack.h>

#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/grouped_assign.h>
#include <cuvs/neighbors/grouped_assign.hpp>

#include "../core/exceptions.hpp"
#include "../core/interop.hpp"

namespace {
using index_t = cuvs::neighbors::grouped_assign::index;

void expect_f32(const DLTensor& t, const char* what)
{
  RAFT_EXPECTS(t.dtype.code == kDLFloat && t.dtype.bits == 32, "%s must be float32", what);
}
}  // namespace

extern "C" cuvsError_t cuvsGroupedAssignIndexCreate(cuvsGroupedAssignIndex_t* index)
{
  return cuvs::core::translate_exceptions([=] { *index = new cuvsGroupedAssignIndex{}; });
}

extern "C" cuvsError_t cuvsGroupedAssignIndexDestroy(cuvsGroupedAssignIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    delete reinterpret_cast<index_t*>(index->addr);
    delete index;
  });
}

extern "C" cuvsError_t cuvsGroupedAssignBuild(cuvsResources_t res,
                                              DLManagedTensor* centroids,
                                              cuvsDistanceType metric,
                                              uint32_t n_lists,
                                              uint32_t n_probes,
                                              uint32_t kmeans_n_iters,
                                              uint32_t tile_align,
                                              cuvsGroupedAssignIndex_t index)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(centroids->dl_tensor),
                 "centroids should have device compatible memory");
    expect_f32(centroids->dl_tensor, "centroids");
    cuvs::neighbors::grouped_assign::index_params params;
    params.n_lists        = n_lists;
    params.n_probes       = n_probes;
    params.kmeans_n_iters = kmeans_n_iters;
    params.tile_align     = tile_align;
    params.metric         = static_cast<cuvs::distance::DistanceType>((int)metric);
    using mdspan_type = raft::device_matrix_view<const float, int64_t, raft::row_major>;
    auto mds          = cuvs::core::from_dlpack<mdspan_type>(centroids);
    auto built        = new index_t(cuvs::neighbors::grouped_assign::build(*res_ptr, params, mds));
    delete reinterpret_cast<index_t*>(index->addr);
    index->addr = reinterpret_cast<uintptr_t>(built);
  });
}

extern "C" cuvsError_t cuvsGroupedAssign(cuvsResources_t res,
                                         cuvsGroupedAssignIndex_t index,
                                         DLManagedTensor* queries,
                                         DLManagedTensor* labels)
{
  return cuvs::core::translate_exceptions([=] {
    auto res_ptr = reinterpret_cast<raft::resources*>(res);
    RAFT_EXPECTS(index->addr != 0, "grouped-assign index is not built");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(queries->dl_tensor),
                 "queries should have device compatible memory");
    RAFT_EXPECTS(cuvs::core::is_dlpack_device_compatible(labels->dl_tensor),
                 "labels should have device compatible memory");
    expect_f32(queries->dl_tensor, "queries");
    auto& l = labels->dl_tensor;
    RAFT_EXPECTS(l.dtype.code == kDLUInt && l.dtype.bits == 32, "labels must be uint32");
    using qspan = raft::device_matrix_view<const float, int64_t, raft::row_major>;
    auto q      = cuvs::core::from_dlpack<qspan>(queries);
    auto lab    = raft::make_device_vector_view<uint32_t, int64_t>(
      reinterpret_cast<uint32_t*>(static_cast<char*>(l.data) + l.byte_offset), l.shape[0]);
    cuvs::neighbors::grouped_assign::assign(*res_ptr, *reinterpret_cast<index_t*>(index->addr), q, lab);
  });
}
