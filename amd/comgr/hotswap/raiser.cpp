//===- raiser.cpp - Hotswap MC -> LLVM IR raiser scaffolding --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raiser.hpp"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/TargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace transpiler {

namespace {

constexpr llvm::StringLiteral AMDGPUTriple = "amdgcn-amd-amdhsa";

// Reject obviously-bad inputs before constructing IR. Mirrors the
// preconditions the full pipeline enforces in subsequent commits.
//
// AGENT_CONVENTIONS.md §1 calls for `COMGR::parseTargetIdentifier` here,
// but that helper currently lives behind the comgr-metadata layer in
// `src/comgr.cpp` and is not reachable from the hotswap subproject. As a
// stop-gap that satisfies §1's "Comgr first, LLVM second" hierarchy,
// validate the AMDGPU processor name through the LLVM target-parser API
// listed in §1 ("Reuse existing LLVM APIs second").
RaiseFailure validateInputs(llvm::StringRef SourceISA,
                            llvm::StringRef KernelName,
                            const KernelMeta &Meta) {
  RaiseFailure F;
  if (SourceISA.empty()) {
    F.reason = RaiseFailureReason::BadInput;
    F.detail = "source ISA string is empty";
    return F;
  }
  // The disassembler-facing identifier is `<arch>-<vendor>-<os>-<env>-<gfx>`;
  // `parseArchAMDGCN` inspects the trailing component.
  llvm::StringRef GfxName = SourceISA.rsplit('-').second;
  if (GfxName.empty()) {
    GfxName = SourceISA;
  }
  if (llvm::AMDGPU::parseArchAMDGCN(GfxName) == llvm::AMDGPU::GK_NONE) {
    F.reason = RaiseFailureReason::BadInput;
    F.detail =
        ("source ISA '" + SourceISA + "' does not name an AMDGPU GPU").str();
    return F;
  }
  if (KernelName.empty()) {
    F.reason = RaiseFailureReason::BadInput;
    F.detail = "kernel name is empty";
    return F;
  }
  if (!Meta.hasKernelDescriptor) {
    F.reason = RaiseFailureReason::BadInput;
    F.detail = ("kernel '" + KernelName + "' has no parsed kernel descriptor")
                   .str();
    return F;
  }
  return F;
}

} // namespace

RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> /*TextBytes*/,
                      llvm::StringRef SourceISA,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      uint64_t /*KernelOffset*/,
                      llvm::StringRef /*CompilationTargetISA*/) {
  using namespace llvm;

  RaiseResult Result;
  Result.failure = validateInputs(SourceISA, KernelName, Meta);
  if (Result.failure.hasFailed()) {
    return Result;
  }

  Result.ctx = std::make_unique<LLVMContext>();
  LLVMContext &C = *Result.ctx;
  Result.module = std::make_unique<Module>("transpiler_module", C);
  Module &M = *Result.module;
  M.setTargetTriple(Triple(AMDGPUTriple));

  auto *FuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg=*/false);
  Function *F =
      Function::Create(FuncTy, GlobalValue::ExternalLinkage, KernelName, &M);
  F->setCallingConv(CallingConv::AMDGPU_KERNEL);

  BasicBlock *Entry = BasicBlock::Create(C, "entry", F);
  IRBuilder<> B(Entry);
  B.CreateRetVoid();

  Result.success = true;
  return Result;
}

} // namespace transpiler
