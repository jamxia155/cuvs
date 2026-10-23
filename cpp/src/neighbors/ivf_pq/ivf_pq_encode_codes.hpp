/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cuvs::neighbors::ivf_pq::detail {

/**
 * Alternative PQ-encode paths for `ivf_pq::transform` (PER_SUBSPACE codebooks, pq_bits == 8).
 *
 * Both take the rotated residuals [n_rows, pq_dim * pq_len] (float) and the codebook
 * [pq_dim, pq_len, 256] (float, row-major), and write one code byte per (row, subspace) into
 * `out` (row stride `out_stride` bytes), the same output as `transform_codes_kernel`.
 *
 * Both fail (RAFT_FAIL) if the shape is unsupported: pq_len must be in [1, 32] for `simt`, and a
 * multiple of 8 in [8, 32] for `tensor`.
 */

/**
 * CUDA-core FP32 kernel with the codebook of one subspace staged in shared memory and an
 * 8-row x 8-code register tile per thread. The per-code distance is the same fused-multiply-add
 * chain as the baseline kernel (`sum_k fma(t, t, d)`, t = x - c), so codes only differ from it
 * on exact distance ties.
 */
void ivf_pq_encode_simt(cudaStream_t stream,
                        const float* residuals,
                        int64_t n_rows,
                        uint32_t pq_dim,
                        uint32_t pq_len,
                        const float* pq_centers,
                        uint8_t* out,
                        uint32_t out_stride);

/**
 * Tensor-core kernel: the dot products x.c of each subspace are an [rows, pq_len] x
 * [pq_len, 256] TF32 matrix product (mma.m16n8k8), and the code is the argmin of
 * |c|^2 - 2 x.c. `tf32_passes` is 1 (plain TF32) or 3 (hi/lo split, 3xTF32, close to FP32).
 */
void ivf_pq_encode_tensor(cudaStream_t stream,
                          const float* residuals,
                          int64_t n_rows,
                          uint32_t pq_dim,
                          uint32_t pq_len,
                          const float* pq_centers,
                          uint8_t* out,
                          uint32_t out_stride,
                          int tf32_passes);

}  // namespace cuvs::neighbors::ivf_pq::detail
