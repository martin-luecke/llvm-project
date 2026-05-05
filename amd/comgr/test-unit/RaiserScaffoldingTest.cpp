//===- RaiserScaffoldingTest.cpp - Hotswap transpiler scaffolding test ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Pins the scaffolding contract `raiseToIR` advertises: an empty input
// produces a well-formed `llvm::Module` containing one `AMDGPU_KERNEL`
// function whose body is exactly `ret void`, with the AMDGPU triple set.
// Empty inputs succeed; missing kernel descriptor / malformed ISA inputs
// are rejected with a structured failure.
//
//===----------------------------------------------------------------------===//

#include "hotswap/raiser.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

namespace {

transpiler::KernelMeta makeKernelMeta(llvm::StringRef Name) {
  transpiler::KernelMeta Meta;
  Meta.name = Name.str();
  Meta.hasKernelDescriptor = true;
  return Meta;
}

} // namespace

TEST(RaiserScaffolding, EmptyInputProducesValidModule) {
  transpiler::KernelMeta Meta = makeKernelMeta("kernel");
  transpiler::RaiseResult Result =
      transpiler::raiseToIR({}, "gfx942", "kernel", Meta);

  ASSERT_TRUE(Result.success);
  ASSERT_NE(Result.ctx, nullptr);
  ASSERT_NE(Result.module, nullptr);

  std::string Err;
  llvm::raw_string_ostream ErrStream(Err);
  EXPECT_FALSE(llvm::verifyModule(*Result.module, &ErrStream)) << Err;
}

TEST(RaiserScaffolding, ModuleAdvertisesAMDGPUTriple) {
  transpiler::KernelMeta Meta = makeKernelMeta("kernel");
  transpiler::RaiseResult Result =
      transpiler::raiseToIR({}, "gfx942", "kernel", Meta);

  ASSERT_TRUE(Result.success);
  ASSERT_NE(Result.module, nullptr);
  EXPECT_EQ(Result.module->getTargetTriple().str(), "amdgcn-amd-amdhsa");
}

TEST(RaiserScaffolding, KernelFunctionIsAMDGPUKernelWithRetVoid) {
  transpiler::KernelMeta Meta = makeKernelMeta("kernel");
  transpiler::RaiseResult Result =
      transpiler::raiseToIR({}, "gfx942", "kernel", Meta);

  ASSERT_TRUE(Result.success);
  llvm::Function *Fn = Result.module->getFunction("kernel");
  ASSERT_NE(Fn, nullptr);
  EXPECT_EQ(Fn->getCallingConv(), llvm::CallingConv::AMDGPU_KERNEL);
  ASSERT_EQ(Fn->size(), 1u);
  llvm::BasicBlock &Entry = Fn->getEntryBlock();
  ASSERT_FALSE(Entry.empty());
  EXPECT_TRUE(llvm::isa<llvm::ReturnInst>(Entry.getTerminator()));
}

TEST(RaiserScaffolding, MissingKernelDescriptorIsRejected) {
  transpiler::KernelMeta Meta;
  Meta.name = "kernel";
  Meta.hasKernelDescriptor = false;
  transpiler::RaiseResult Result =
      transpiler::raiseToIR({}, "gfx942", "kernel", Meta);

  EXPECT_FALSE(Result.success);
  EXPECT_TRUE(Result.failure.hasFailed());
}

TEST(RaiserScaffolding, EmptyTargetIsaIsRejected) {
  transpiler::KernelMeta Meta = makeKernelMeta("kernel");
  transpiler::RaiseResult Result =
      transpiler::raiseToIR({}, "", "kernel", Meta);

  EXPECT_FALSE(Result.success);
  EXPECT_TRUE(Result.failure.hasFailed());
}

TEST(RaiserScaffolding, MalformedTargetIsaIsRejected) {
  transpiler::KernelMeta Meta = makeKernelMeta("kernel");
  transpiler::RaiseResult Result =
      transpiler::raiseToIR({}, "not-a-real-isa", "kernel", Meta);

  EXPECT_FALSE(Result.success);
  EXPECT_TRUE(Result.failure.hasFailed());
}
