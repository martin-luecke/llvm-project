//===- hotswap-dispatch.cpp - Checked dispatch rewriting -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-dispatch.h"

#include <cstdint>
#include <limits>

namespace COMGR::hotswap::hsa_tool {

DispatchRewriteError rewriteKernelDispatch(const KernelDispatchTarget &Target,
                                           KernelDispatchPacket &Packet) {
  if (Target.KernelObject == 0 || Target.Scale == 0 ||
      Target.MaxWorkgroupSizeX == 0 || Target.MaxWorkgroupSizeY == 0 ||
      Target.MaxWorkgroupSizeZ == 0 || Target.MaxWorkgroupSize == 0 ||
      Target.MaxGridSizeX == 0 || Target.MaxGridSizeY == 0 ||
      Target.MaxGridSizeZ == 0 || Target.MaxGridSize == 0)
    return DispatchRewriteError::InvalidTarget;
  if (Packet.PrivateSegmentSize < Target.SourcePrivateSegmentSize)
    return DispatchRewriteError::PrivateSegmentTooSmall;
  if (Packet.GroupSegmentSize < Target.SourceGroupSegmentSize)
    return DispatchRewriteError::GroupSegmentTooSmall;

  const uint64_t Private =
      static_cast<uint64_t>(Packet.PrivateSegmentSize -
                            Target.SourcePrivateSegmentSize) +
      Target.TargetPrivateSegmentSize;
  if (Private > std::numeric_limits<uint32_t>::max())
    return DispatchRewriteError::PrivateSegmentOverflow;
  const uint64_t Group = static_cast<uint64_t>(Packet.GroupSegmentSize -
                                               Target.SourceGroupSegmentSize) +
                         Target.TargetGroupSegmentSize;
  if (Group > std::numeric_limits<uint32_t>::max())
    return DispatchRewriteError::GroupSegmentOverflow;

  const uint64_t Workgroup =
      static_cast<uint64_t>(Packet.WorkgroupSizeX) * Target.Scale;
  if (Workgroup > std::numeric_limits<uint16_t>::max())
    return DispatchRewriteError::WorkgroupSizeOverflow;
  if (Workgroup > Target.MaxWorkgroupSizeX)
    return DispatchRewriteError::WorkgroupDimensionExceeded;
  if (Packet.WorkgroupSizeY > Target.MaxWorkgroupSizeY ||
      Packet.WorkgroupSizeZ > Target.MaxWorkgroupSizeZ)
    return DispatchRewriteError::WorkgroupDimensionExceeded;
  const uint64_t WorkgroupTotal =
      Workgroup * Packet.WorkgroupSizeY * Packet.WorkgroupSizeZ;
  if (WorkgroupTotal > Target.MaxWorkgroupSize)
    return DispatchRewriteError::WorkgroupSizeExceeded;
  if (Packet.Grid) {
    const uint64_t Grid = static_cast<uint64_t>(Packet.Grid->X) * Target.Scale;
    if (Grid > std::numeric_limits<uint32_t>::max())
      return DispatchRewriteError::GridSizeOverflow;
    if (Grid > Target.MaxGridSizeX || Packet.Grid->Y > Target.MaxGridSizeY ||
        Packet.Grid->Z > Target.MaxGridSizeZ)
      return DispatchRewriteError::GridDimensionExceeded;
    const uint64_t GridXY = Grid * Packet.Grid->Y;
    if (Packet.Grid->Z != 0 &&
        GridXY > std::numeric_limits<uint64_t>::max() / Packet.Grid->Z)
      return DispatchRewriteError::GridSizeOverflow;
    if (GridXY * Packet.Grid->Z > Target.MaxGridSize)
      return DispatchRewriteError::GridSizeExceeded;
    Packet.Grid->X = static_cast<uint32_t>(Grid);
  }

  Packet.PrivateSegmentSize = static_cast<uint32_t>(Private);
  Packet.GroupSegmentSize = static_cast<uint32_t>(Group);
  Packet.WorkgroupSizeX = static_cast<uint16_t>(Workgroup);
  Packet.KernelObject = Target.KernelObject;
  return DispatchRewriteError::None;
}

const char *dispatchRewriteErrorString(DispatchRewriteError Error) {
  switch (Error) {
  case DispatchRewriteError::None:
    return "none";
  case DispatchRewriteError::InvalidTarget:
    return "invalid translated kernel target";
  case DispatchRewriteError::PrivateSegmentTooSmall:
    return "dispatch private segment is smaller than source kernel";
  case DispatchRewriteError::GroupSegmentTooSmall:
    return "dispatch group segment is smaller than source kernel";
  case DispatchRewriteError::PrivateSegmentOverflow:
    return "adjusted private segment size overflows";
  case DispatchRewriteError::GroupSegmentOverflow:
    return "adjusted group segment size overflows";
  case DispatchRewriteError::WorkgroupSizeOverflow:
    return "scaled workgroup size overflows";
  case DispatchRewriteError::WorkgroupDimensionExceeded:
    return "scaled workgroup x exceeds the physical agent limit";
  case DispatchRewriteError::WorkgroupSizeExceeded:
    return "scaled workgroup size exceeds the physical agent limit";
  case DispatchRewriteError::GridSizeOverflow:
    return "scaled grid size overflows";
  case DispatchRewriteError::GridDimensionExceeded:
    return "scaled grid dimension exceeds the physical agent limit";
  case DispatchRewriteError::GridSizeExceeded:
    return "scaled grid size exceeds the physical agent limit";
  }
  return "unknown dispatch rewrite error";
}

} // namespace COMGR::hotswap::hsa_tool
