//===- kernarg-layout.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernarg-layout.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>

using namespace llvm;

namespace COMGR::hotswap {
namespace {

uint64_t implicitArgsBase(ArrayRef<KernelArgMeta> Args) {
  uint64_t MaxEnd = 0;
  for (const KernelArgMeta &Arg : Args) {
    if (StringRef(Arg.ValueKind).starts_with("hidden_"))
      continue;
    uint64_t End = static_cast<uint64_t>(Arg.Offset) + Arg.Size;
    if (End > MaxEnd)
      MaxEnd = End;
  }
  return (MaxEnd + 7u) & ~uint64_t(7u);
}

struct StandardHiddenArg {
  int RelOffset;
  int Size;
  SourceHiddenArgKind Kind;
  StringRef ValueKind;
};

SourceHiddenArgByte classifyStandardHiddenArgByte(ArrayRef<KernelArgMeta> Args,
                                                  int ByteOffset) {
  uint64_t Base = implicitArgsBase(Args);
  if (ByteOffset < 0 || static_cast<uint64_t>(ByteOffset) < Base)
    return {};

  int Rel = ByteOffset - static_cast<int>(Base);
  static constexpr StandardHiddenArg StandardHiddenArgs[] = {
      {0, 4, SourceHiddenArgKind::HiddenBlockCountX, "hidden_block_count_x"},
      {4, 4, SourceHiddenArgKind::HiddenBlockCountY, "hidden_block_count_y"},
      {8, 4, SourceHiddenArgKind::HiddenBlockCountZ, "hidden_block_count_z"},
      {12, 2, SourceHiddenArgKind::HiddenGroupSizeX, "hidden_group_size_x"},
      {14, 2, SourceHiddenArgKind::HiddenGroupSizeY, "hidden_group_size_y"},
      {16, 2, SourceHiddenArgKind::HiddenGroupSizeZ, "hidden_group_size_z"},
      {18, 2, SourceHiddenArgKind::HiddenRemainderX, "hidden_remainder_x"},
      {20, 2, SourceHiddenArgKind::HiddenRemainderY, "hidden_remainder_y"},
      {22, 2, SourceHiddenArgKind::HiddenRemainderZ, "hidden_remainder_z"},
      {24, 8, SourceHiddenArgKind::HiddenReservedZero,
       "hidden_tool_correlation_id_reserved"},
      {32, 8, SourceHiddenArgKind::HiddenReservedZero, "hidden_reserved_32"},
      {40, 8, SourceHiddenArgKind::HiddenGlobalOffsetX, "hidden_global_offset_x"},
      {48, 8, SourceHiddenArgKind::HiddenGlobalOffsetY, "hidden_global_offset_y"},
      {56, 8, SourceHiddenArgKind::HiddenGlobalOffsetZ, "hidden_global_offset_z"},
      {64, 2, SourceHiddenArgKind::HiddenGridDims, "hidden_grid_dims"},
      {66, 6, SourceHiddenArgKind::HiddenReservedZero, "hidden_reserved_66"},
      {72, 8, SourceHiddenArgKind::UnsupportedHidden, "hidden_printf_buffer"},
      {80, 8, SourceHiddenArgKind::UnsupportedHidden, "hidden_hostcall_buffer"},
      {88, 8, SourceHiddenArgKind::UnsupportedHidden,
       "hidden_multigrid_sync_arg"},
      {96, 8, SourceHiddenArgKind::UnsupportedHidden, "hidden_heap_v1"},
      {104, 8, SourceHiddenArgKind::UnsupportedHidden, "hidden_default_queue"},
      {112, 8, SourceHiddenArgKind::UnsupportedHidden,
       "hidden_completion_action"},
      {120, 4, SourceHiddenArgKind::UnsupportedHidden,
       "hidden_dynamic_lds_size"},
      {124, 68, SourceHiddenArgKind::HiddenReservedZero, "hidden_reserved_124"},
      {192, 4, SourceHiddenArgKind::HiddenPrivateBase, "hidden_private_base"},
      {196, 4, SourceHiddenArgKind::HiddenSharedBase, "hidden_shared_base"},
      {200, 8, SourceHiddenArgKind::HiddenQueuePtr, "hidden_queue_ptr"},
      {208, 48, SourceHiddenArgKind::HiddenReservedZero, "hidden_reserved_208"},
  };

  for (const StandardHiddenArg &Arg : StandardHiddenArgs) {
    int ArgEnd = Arg.RelOffset + Arg.Size;
    if (Rel < Arg.RelOffset || Rel >= ArgEnd)
      continue;
    SourceHiddenArgByte Result;
    Result.Kind = Arg.Kind;
    Result.ValueKind = Arg.ValueKind;
    Result.ArgOffset = static_cast<int>(Base) + Arg.RelOffset;
    Result.ByteOffset = ByteOffset;
    return Result;
  }
  return {};
}

} // namespace

std::optional<SourceHiddenArgByte>
classifySourceHiddenArgByte(ArrayRef<KernelArgMeta> Args, int ByteOffset) {
  if (ByteOffset < 0)
    return std::nullopt;
  uint32_t Offset = static_cast<uint32_t>(ByteOffset);

  for (const KernelArgMeta &Arg : Args) {
    if (Offset < Arg.Offset || Offset >= Arg.Offset + Arg.Size)
      continue;

    StringRef Kind(Arg.ValueKind);
    if (!Kind.starts_with("hidden_"))
      return std::nullopt;

    SourceHiddenArgByte Result;
    Result.ValueKind = Kind;
    Result.ArgOffset = static_cast<int>(Arg.Offset);
    Result.ByteOffset = ByteOffset;
    if (Kind == "hidden_block_count_x")
      Result.Kind = SourceHiddenArgKind::HiddenBlockCountX;
    else if (Kind == "hidden_block_count_y")
      Result.Kind = SourceHiddenArgKind::HiddenBlockCountY;
    else if (Kind == "hidden_block_count_z")
      Result.Kind = SourceHiddenArgKind::HiddenBlockCountZ;
    else if (Kind == "hidden_group_size_x")
      Result.Kind = SourceHiddenArgKind::HiddenGroupSizeX;
    else if (Kind == "hidden_group_size_y")
      Result.Kind = SourceHiddenArgKind::HiddenGroupSizeY;
    else if (Kind == "hidden_group_size_z")
      Result.Kind = SourceHiddenArgKind::HiddenGroupSizeZ;
    else if (Kind == "hidden_remainder_x")
      Result.Kind = SourceHiddenArgKind::HiddenRemainderX;
    else if (Kind == "hidden_remainder_y")
      Result.Kind = SourceHiddenArgKind::HiddenRemainderY;
    else if (Kind == "hidden_remainder_z")
      Result.Kind = SourceHiddenArgKind::HiddenRemainderZ;
    else if (Kind == "hidden_grid_dims")
      Result.Kind = SourceHiddenArgKind::HiddenGridDims;
    else if (Kind == "hidden_global_offset_x")
      Result.Kind = SourceHiddenArgKind::HiddenGlobalOffsetX;
    else if (Kind == "hidden_global_offset_y")
      Result.Kind = SourceHiddenArgKind::HiddenGlobalOffsetY;
    else if (Kind == "hidden_global_offset_z")
      Result.Kind = SourceHiddenArgKind::HiddenGlobalOffsetZ;
    else if (Kind == "hidden_private_base")
      Result.Kind = SourceHiddenArgKind::HiddenPrivateBase;
    else if (Kind == "hidden_shared_base")
      Result.Kind = SourceHiddenArgKind::HiddenSharedBase;
    else if (Kind == "hidden_queue_ptr")
      Result.Kind = SourceHiddenArgKind::HiddenQueuePtr;
    else
      Result.Kind = SourceHiddenArgKind::UnsupportedHidden;
    return Result;
  }
  return std::nullopt;
}

} // namespace COMGR::hotswap
