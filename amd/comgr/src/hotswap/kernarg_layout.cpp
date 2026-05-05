//===- kernarg_layout.cpp - Hotswap transpiler ----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "kernarg_layout.h"

#include "llvm/ADT/StringRef.h"

using namespace llvm;

namespace COMGR::hotswap {

PreloadedHiddenKernargDword classifyPreloadedHiddenKernargDword(
    ArrayRef<KernelArgMeta> Args, int ByteOffset) {
  for (const KernelArgMeta &Arg : Args) {
    if (Arg.Offset != ByteOffset)
      continue;
    StringRef Kind(Arg.ValueKind);
    if (!Kind.starts_with("hidden_"))
      return PreloadedHiddenKernargDword::NotHidden;
    if (Kind == "hidden_block_count_x")
      return PreloadedHiddenKernargDword::HiddenBlockCountX;
    if (Kind == "hidden_block_count_y")
      return PreloadedHiddenKernargDword::HiddenBlockCountY;
    if (Kind == "hidden_block_count_z")
      return PreloadedHiddenKernargDword::HiddenBlockCountZ;
    return PreloadedHiddenKernargDword::UnsupportedHidden;
  }
  return PreloadedHiddenKernargDword::NotHidden;
}

} // namespace COMGR::hotswap
