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

#ifdef __cplusplus
}
#endif
