//===- raiser.h - Hotswap MC -> LLVM IR raiser entry point --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISER_H
#define HOTSWAP_TRANSPILER_RAISER_H

#include "code_object_utils.h"
#include "raise_failure.h"

#include "llvm/ADT/StringRef.h"

#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Module;
  int TotalCount = 0;
  // Structured failure description. `failure.reason == None` iff `success`.
  RaiseFailure Failure;
  bool Success = false;
};

// Raise a kernel named `KernelName` from the disassembled `TextBytes` of the
// AMDGPU code object whose source ISA is `SourceISA`. `Meta` carries the
// MsgPack-derived per-kernel metadata. `KernelOffset` is the byte offset of
// the kernel's code-object entry point inside `TextBytes`.
// `CompilationTargetISA` (defaults to the source ISA) selects the AMDGPU
// codegen target the lifted IR will be compiled for.
RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                      llvm::StringRef SourceISA,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      uint64_t KernelOffset = 0,
                      llvm::StringRef CompilationTargetISA = "");

} // namespace COMGR::hotswap

#endif
