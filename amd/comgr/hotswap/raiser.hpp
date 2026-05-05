//===- raiser.hpp - Hotswap MC -> LLVM IR raiser entry point --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISER_HPP
#define HOTSWAP_TRANSPILER_RAISER_HPP

#include "code_object_utils.hpp"
#include "raise_failure.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace transpiler {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<llvm::Module> module;
  // Structured failure description. `failure.reason == None` iff `success`.
  RaiseFailure failure;
  bool success = false;
};

// Raise the kernel `KernelName` from the disassembled `TextBytes` of the
// AMDGPU code object whose source ISA is `SourceISA`. `Meta` carries the
// MsgPack-derived per-kernel metadata. The scaffolding implementation
// emits a `ret void` placeholder and refuses inputs the full pipeline
// would also refuse: missing kernel descriptor, empty kernel name, and
// `SourceISA` strings that don't parse via `llvm::AMDGPU::parseArchAMDGCN`.
RaiseResult raiseToIR(llvm::ArrayRef<uint8_t> TextBytes,
                      llvm::StringRef SourceISA,
                      llvm::StringRef KernelName,
                      const KernelMeta &Meta,
                      uint64_t KernelOffset = 0,
                      llvm::StringRef CompilationTargetISA = "");

} // namespace transpiler

#endif
