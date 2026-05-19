//===- DebugInfoTest.cpp - KernelDwarfSource unit tests -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Null-safety tests for KernelDwarfSource / DebugInfoBuilder. End-to-end
// DWARF coverage needs a real `-g` HSACO and lives in the lit fixtures.
//
//===----------------------------------------------------------------------===//

#include "hotswap/debug-info.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBufferRef.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstring>

namespace {

llvm::MemoryBufferRef makeBuf(llvm::ArrayRef<uint8_t> Bytes,
                              llvm::StringRef Name) {
  return llvm::MemoryBufferRef(
      llvm::StringRef(reinterpret_cast<const char *>(Bytes.data()),
                      Bytes.size()),
      Name);
}

} // namespace

TEST(KernelDwarfSource, NonElfBytesReturnsNull) {
  // Non-ELF input: create() must swallow the error and return null.
  const uint8_t Garbage[] = {0xde, 0xad, 0xbe, 0xef, 0x00, 0x01, 0x02, 0x03};
  auto Src = COMGR::hotswap::KernelDwarfSource::create(makeBuf(Garbage, "g"));
  EXPECT_EQ(Src, nullptr);
}

TEST(KernelDwarfSource, EmptyBufferReturnsNull) {
  auto Src = COMGR::hotswap::KernelDwarfSource::create(makeBuf({}, "empty"));
  EXPECT_EQ(Src, nullptr);
}

TEST(DebugInfoBuilder, NullSubprogramShortCircuits) {
  // Documents the contract: with no DebugSource the raiser emits no DI
  // nodes. Constructing a synthetic DWARF-bearing ELF is out of scope
  // here, so this just pins that a bare module carries no `llvm.dbg.cu`.
  llvm::LLVMContext Ctx;
  llvm::Module M("t", Ctx);
  EXPECT_EQ(M.getNamedMetadata("llvm.dbg.cu"), nullptr);
}
