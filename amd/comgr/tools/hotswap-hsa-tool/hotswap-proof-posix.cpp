//===- hotswap-proof-posix.cpp - POSIX proof log support -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-proof.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/types.h>
#include <unistd.h>

namespace COMGR::hotswap::hsa_tool {

std::string jsonEscape(llvm::StringRef Value) {
  std::string ValidUtf8;
  if (!llvm::json::isUTF8(Value)) {
    ValidUtf8 = llvm::json::fixUTF8(Value);
    Value = ValidUtf8;
  }
  constexpr unsigned FirstPrintableAscii = 0x20;
  std::string Result;
  Result.reserve(Value.size());
  for (const unsigned char C : Value.bytes()) {
    switch (C) {
    case '\\':
      Result += "\\\\";
      break;
    case '"':
      Result += "\\\"";
      break;
    case '\b':
      Result += "\\b";
      break;
    case '\f':
      Result += "\\f";
      break;
    case '\n':
      Result += "\\n";
      break;
    case '\r':
      Result += "\\r";
      break;
    case '\t':
      Result += "\\t";
      break;
    default:
      if (C < FirstPrintableAscii) {
        Result += "\\u00";
        Result += llvm::hexdigit(C >> 4, /*LowerCase=*/true);
        Result += llvm::hexdigit(C & 0xf, /*LowerCase=*/true);
      } else {
        Result += static_cast<char>(C);
      }
      break;
    }
  }
  return Result;
}

namespace {

enum class FileLock { Exclusive, Unlock };

bool setFileLock(int File, FileLock Type) {
  struct flock Lock{};
  Lock.l_type = Type == FileLock::Exclusive ? F_WRLCK : F_UNLCK;
  Lock.l_whence = SEEK_SET;
  while (fcntl(File, F_SETLKW, &Lock) == -1) {
    if (errno != EINTR)
      return false;
  }
  return true;
}

bool writeAll(int File, llvm::StringRef Contents) {
  while (!Contents.empty()) {
    const size_t Chunk =
        std::min(Contents.size(),
                 static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
    const ssize_t Count = write(File, Contents.data(), Chunk);
    if (Count > 0) {
      Contents = Contents.drop_front(static_cast<size_t>(Count));
      continue;
    }
    if (Count < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}

} // namespace

bool appendProofLine(const ProofLine &Record) {
  if (Record.Path.empty() || Record.Path.contains('\0'))
    return false;
  std::string NullTerminatedPath = Record.Path.str();
  const int File = open(NullTerminatedPath.c_str(),
                        O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0666);
  if (File == -1)
    return false;

  std::string Line = "{\"pid\":" + std::to_string(Record.ProcessId) + ",";
  Line.append(Record.Fields.data(), Record.Fields.size());
  Line += "}\n";

  // O_APPEND makes the file-offset update atomic. The advisory write lock also
  // spans retries after a partial write, so cooperating tool processes cannot
  // interleave fragments of two proof records.
  const bool Locked = setFileLock(File, FileLock::Exclusive);
  const bool Written = Locked && writeAll(File, Line);
  const bool Unlocked = Locked && setFileLock(File, FileLock::Unlock);
  const bool Closed = close(File) == 0;
  return Written && Unlocked && Closed;
}

} // namespace COMGR::hotswap::hsa_tool
