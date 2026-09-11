/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/c_api.h>
#include "../core/exceptions.hpp"
#include <cuvs/util/file_io.h>
#include <cuvs/util/file_io.hpp>

#include <fcntl.h>

#include <memory>
#include <utility>

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

extern "C" cuvsError_t cuvsReadLargeFileAsync(const char* path,
                                              void* dest_ptr,
                                              size_t total_bytes,
                                              uint64_t file_offset,
                                              cudaStream_t stream,
                                              cuvsGdsReadFuture_t* future_out)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(path != nullptr, "cuvsReadLargeFileAsync: path must not be null");
    RAFT_EXPECTS(future_out != nullptr, "cuvsReadLargeFileAsync: future_out must not be null");

    cuvs::util::file_descriptor fd(path, O_RDONLY);
    auto future = cuvs::util::read_large_file_async(
      fd, dest_ptr, total_bytes, file_offset, reinterpret_cast<void*>(stream));
    *future_out = new cuvs::util::GdsReadFuture(std::move(future));
  });
}

extern "C" cuvsError_t cuvsFinishReadLargeFileAsync(cuvsGdsReadFuture_t future, size_t total_bytes)
{
  return cuvs::core::translate_exceptions([=] {
    RAFT_EXPECTS(future != nullptr, "cuvsFinishReadLargeFileAsync: future must not be null");

    std::unique_ptr<cuvs::util::GdsReadFuture> owned(
      static_cast<cuvs::util::GdsReadFuture*>(future));
    const size_t bytes_read = cuvs::util::finish_read_large_file_async(std::move(*owned));
    RAFT_EXPECTS(bytes_read == total_bytes,
                 "cuvsFinishReadLargeFileAsync: short read (expected %zu bytes, got %zu)",
                 total_bytes,
                 bytes_read);
  });
}
