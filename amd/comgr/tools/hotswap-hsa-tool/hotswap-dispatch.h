//===- hotswap-dispatch.h - Checked dispatch rewriting ----------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TOOLS_HOTSWAP_HSA_TOOL_DISPATCH_H
#define COMGR_TOOLS_HOTSWAP_HSA_TOOL_DISPATCH_H

#include <cstdint>
#include <optional>

namespace COMGR::hotswap::hsa_tool {

struct KernelDispatchTarget {
  uint64_t KernelObject = 0;
  uint32_t SourcePrivateSegmentSize = 0;
  uint32_t TargetPrivateSegmentSize = 0;
  uint32_t SourceGroupSegmentSize = 0;
  uint32_t TargetGroupSegmentSize = 0;
  uint32_t Scale = 1;
  uint32_t MaxWorkgroupSizeX = 0;
  uint32_t MaxWorkgroupSizeY = 0;
  uint32_t MaxWorkgroupSizeZ = 0;
  uint32_t MaxWorkgroupSize = 0;
  uint32_t MaxGridSizeX = 0;
  uint32_t MaxGridSizeY = 0;
  uint32_t MaxGridSizeZ = 0;
  uint64_t MaxGridSize = 0;
};

struct DispatchGrid {
  uint32_t X = 0;
  uint32_t Y = 0;
  uint32_t Z = 0;
};

struct KernelDispatchPacket {
  uint64_t KernelObject = 0;
  uint32_t PrivateSegmentSize = 0;
  uint32_t GroupSegmentSize = 0;
  uint16_t WorkgroupSizeX = 0;
  uint16_t WorkgroupSizeY = 0;
  uint16_t WorkgroupSizeZ = 0;
  std::optional<DispatchGrid> Grid;
};

enum class DispatchRewriteError {
  None,
  InvalidTarget,
  PrivateSegmentTooSmall,
  GroupSegmentTooSmall,
  PrivateSegmentOverflow,
  GroupSegmentOverflow,
  WorkgroupSizeOverflow,
  WorkgroupDimensionExceeded,
  WorkgroupSizeExceeded,
  GridSizeOverflow,
  GridDimensionExceeded,
  GridSizeExceeded,
};

DispatchRewriteError rewriteKernelDispatch(const KernelDispatchTarget &Target,
                                           KernelDispatchPacket &Packet);

const char *dispatchRewriteErrorString(DispatchRewriteError Error);

} // namespace COMGR::hotswap::hsa_tool

#endif
