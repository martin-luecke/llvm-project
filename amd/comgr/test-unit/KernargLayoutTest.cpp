//===- kernarg_layout_test.cpp - kernarg_layout unit tests ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/kernarg_layout.h"

#include "gtest/gtest.h"

#include <vector>

using COMGR::hotswap::KernelArgMeta;
using COMGR::hotswap::PreloadedHiddenKernargDword;
using COMGR::hotswap::classifyPreloadedHiddenKernargDword;

namespace {
KernelArgMeta makeArg(const char *Name, int Offset, int Size,
                      const char *ValueKind) {
  KernelArgMeta Arg;
  Arg.Name = Name;
  Arg.Offset = Offset;
  Arg.Size = Size;
  Arg.ValueKind = ValueKind;
  return Arg;
}
} // namespace

TEST(KernargLayout, ClassifiesPreloadedHiddenBlockCounts) {
  std::vector<KernelArgMeta> Args = {
      makeArg("out", 0, 8, "global_buffer"),
      makeArg("grid_x", 48, 4, "hidden_block_count_x"),
      makeArg("grid_y", 52, 4, "hidden_block_count_y"),
      makeArg("grid_z", 56, 4, "hidden_block_count_z"),
  };

  EXPECT_EQ(classifyPreloadedHiddenKernargDword(Args, 48),
            PreloadedHiddenKernargDword::HiddenBlockCountX);
  EXPECT_EQ(classifyPreloadedHiddenKernargDword(Args, 52),
            PreloadedHiddenKernargDword::HiddenBlockCountY);
  EXPECT_EQ(classifyPreloadedHiddenKernargDword(Args, 56),
            PreloadedHiddenKernargDword::HiddenBlockCountZ);
}

TEST(KernargLayout, ClassifiesUnsupportedPreloadedHiddenKinds) {
  std::vector<KernelArgMeta> Args = {
      makeArg("hostcall", 64, 8, "hidden_hostcall_buffer"),
  };

  EXPECT_EQ(classifyPreloadedHiddenKernargDword(Args, 64),
            PreloadedHiddenKernargDword::UnsupportedHidden);
}

TEST(KernargLayout, NonHiddenAndMissingOffsetsAreNotHidden) {
  std::vector<KernelArgMeta> Args = {
      makeArg("n", 24, 4, "by_value"),
  };

  EXPECT_EQ(classifyPreloadedHiddenKernargDword(Args, 24),
            PreloadedHiddenKernargDword::NotHidden);
  EXPECT_EQ(classifyPreloadedHiddenKernargDword(Args, 28),
            PreloadedHiddenKernargDword::NotHidden);
}
