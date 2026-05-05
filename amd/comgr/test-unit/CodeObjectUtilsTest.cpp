//===- code_object_utils_test.cpp - code_object_utils unit tests ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/code_object_utils.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <string>

TEST(CodeObjectUtils, KernelSymbolOffsetMalformedElfReturnsError) {
  uint8_t Garbage[] = {0x7f, 'E', 'L', 'F', 0, 0, 0, 0};

  llvm::Expected<uint64_t> Offset =
      COMGR::hotswap::findKernelSymbolOffset(Garbage, "missing_kernel");

  ASSERT_FALSE(static_cast<bool>(Offset));
  std::string Message = llvm::toString(Offset.takeError());
  EXPECT_NE(Message.find("findKernelSymbolOffset: Failed to parse ELF"),
            std::string::npos);
}
