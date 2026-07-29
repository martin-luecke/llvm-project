//===- hotswap-platform-io-posix.cpp - POSIX file I/O ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-platform-io.h"

#include "llvm/ADT/ArrayRef.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace COMGR::hotswap::hsa_tool {

Bytes readFile(hsa_file_t File, uint64_t Offset, uint64_t Size) {
  if (Size == 0 ||
      Offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
      Size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      Size - 1 >
          static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - Offset)
    return {};

  Bytes Result(new (std::nothrow) std::vector<uint8_t>);
  if (!Result || Size > Result->max_size())
    return {};
  Result->resize(static_cast<size_t>(Size));

  size_t Done = 0;
  while (Done != Result->size()) {
    llvm::MutableArrayRef<uint8_t> Remaining(*Result);
    Remaining = Remaining.drop_front(Done);
    const size_t Chunk =
        std::min(Remaining.size(),
                 static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t Count =
        pread(File, Remaining.data(), Chunk,
              static_cast<off_t>(Offset + static_cast<uint64_t>(Done)));
    if (Count > 0) {
      Done += static_cast<size_t>(Count);
      continue;
    }
    if (Count < 0 && errno == EINTR)
      continue;
    return {};
  }
  return Result;
}

Bytes readWholeFile(hsa_file_t File) {
  struct stat Stat{};
  if (fstat(File, &Stat) != 0 || Stat.st_size <= 0)
    return {};
  return readFile(File, 0, static_cast<uint64_t>(Stat.st_size));
}

uint64_t processId() { return static_cast<uint64_t>(getpid()); }

} // namespace COMGR::hotswap::hsa_tool
