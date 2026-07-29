//===- hotswap-platform-io.h - Platform file I/O ----------------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TOOLS_HOTSWAP_HSA_TOOL_PLATFORM_IO_H
#define COMGR_TOOLS_HOTSWAP_HSA_TOOL_PLATFORM_IO_H

#include <cstddef>
#include <cstdint>
#include <hsa.h>
#include <memory>
#include <vector>

namespace COMGR::hotswap::hsa_tool {

using Bytes = std::shared_ptr<std::vector<uint8_t>>;

Bytes readFile(hsa_file_t File, uint64_t Offset, uint64_t Size);
Bytes readWholeFile(hsa_file_t File);
uint64_t processId();

} // namespace COMGR::hotswap::hsa_tool

#endif
