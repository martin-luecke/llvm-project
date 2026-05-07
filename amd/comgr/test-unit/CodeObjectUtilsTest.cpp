//===- code_object_utils_test.cpp - code_object_utils unit tests ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/code_object_utils.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <vector>

namespace {

// Minimal "ELF-shaped garbage" that does not parse as a valid AMDGPU
// code object: the magic bytes are correct but everything else is zero.
std::vector<uint8_t> garbageElf() {
  return {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};
}

} // namespace

TEST(CodeObjectUtils, EmptyDataParsesAsEmpty) {
  std::vector<uint8_t> empty;
  EXPECT_TRUE(COMGR::hotswap::listKernelNames(empty).empty());
  EXPECT_FALSE(COMGR::hotswap::extractTextSection(empty).Valid);
  EXPECT_TRUE(COMGR::hotswap::detectIsaFromElf(empty).empty());
}

TEST(CodeObjectUtils, MalformedElfYieldsNoKernels) {
  auto data = garbageElf();
  EXPECT_TRUE(COMGR::hotswap::listKernelNames(data).empty());
  EXPECT_FALSE(COMGR::hotswap::extractTextSection(data).Valid);
  EXPECT_TRUE(COMGR::hotswap::detectIsaFromElf(data).empty());
}

TEST(CodeObjectUtils, MissingKernelMetaIsDefaulted) {
  auto data = garbageElf();
  COMGR::hotswap::KernelMeta meta =
      COMGR::hotswap::extractKernelMeta(data, "missing_kernel");
  EXPECT_FALSE(meta.HasKernelDescriptor);
  EXPECT_EQ(meta.KernargSegmentSize, 0);
}
