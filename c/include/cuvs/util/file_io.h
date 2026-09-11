/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuvs/core/c_api.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Read a region of a file directly into a host or device buffer.
 *
 * Uses GPUDirect Storage (via KvikIO) when @p dest_ptr is device memory on a GDS-capable
 * system, and KvikIO's compatible I/O backend (POSIX pread, optionally O_DIRECT) otherwise.
 * @p dest_ptr's memory kind (host vs. device) is detected automatically. This is a random-access
 * read: @p file_offset may be any valid offset into the file, and successive calls need not be
 * sequential or ordered.
 *
 * @param[in]  path         Null-terminated path to the file to read from.
 * @param[out] dest_ptr     Destination buffer (host or device), at least @p total_bytes long.
 * @param[in]  total_bytes  Number of bytes to read.
 * @param[in]  file_offset  Byte offset into the file to start reading from.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsReadLargeFile(const char* path,
                                          void* dest_ptr,
                                          size_t total_bytes,
                                          uint64_t file_offset);

/**
 * @brief Opaque handle to a pending, stream-ordered GDS read started by
 * ::cuvsReadLargeFileAsync.
 */
typedef void* cuvsGdsReadFuture_t;

/**
 * @brief Begin a stream-ordered, random-access read of a region of a file directly into a device
 * buffer.
 *
 * Enqueues the read onto @p stream and returns immediately without blocking the calling thread,
 * unlike ::cuvsReadLargeFile. @p dest_ptr must be device memory. The returned future must be kept
 * alive -- do not call ::cuvsFinishReadLargeFileAsync on it -- until @p stream has been
 * synchronized past this read (e.g. via `cudaStreamSynchronize`); finishing it earlier is
 * undefined behavior.
 *
 * @param[in]  path         Null-terminated path to the file to read from.
 * @param[out] dest_ptr     Destination buffer (device memory), at least @p total_bytes long.
 * @param[in]  total_bytes  Number of bytes to read.
 * @param[in]  file_offset  Byte offset into the file to start reading from.
 * @param[in]  stream       CUDA stream to order the read on.
 * @param[out] future_out   Receives the pending-read handle on success.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsReadLargeFileAsync(const char* path,
                                               void* dest_ptr,
                                               size_t total_bytes,
                                               uint64_t file_offset,
                                               cudaStream_t stream,
                                               cuvsGdsReadFuture_t* future_out);

/**
 * @brief Complete a pending read started by ::cuvsReadLargeFileAsync.
 *
 * Must only be called after the CUDA stream the read was enqueued on has been synchronized past
 * it -- calling this earlier is undefined behavior. Consumes and releases @p future.
 *
 * @param[in] future       The pending read to complete.
 * @param[in] total_bytes  The number of bytes the read was started with; verified against the
 *                         number of bytes actually read.
 * @return cuvsError_t
 */
CUVS_EXPORT cuvsError_t cuvsFinishReadLargeFileAsync(cuvsGdsReadFuture_t future,
                                                     size_t total_bytes);

#ifdef __cplusplus
}
#endif
