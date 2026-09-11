/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include "../core/exceptions.hpp"
#include <cuvs/util/file_io.h>
#include <cuvs/util/file_io.hpp>

#include <fcntl.h>

extern "C" cuvsError_t cuvsReadLargeFile(const char* path,
                                         void* dest_ptr,
                                         size_t total_bytes,
                                         uint64_t file_offset)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(path != nullptr, "cuvsReadLargeFile: path must not be null");

    cuvs::util::file_descriptor fd(path, O_RDONLY);
    cuvs::util::read_large_file(fd, dest_ptr, total_bytes, file_offset);
  });
}
