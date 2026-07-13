//===- raiser.h - Hotswap MC -> LLVM IR raiser entry point --------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISER_H
#define HOTSWAP_TRANSPILER_RAISER_H

#include "code-object-utils.h"
#include "raise-failure.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseResult {
  std::unique_ptr<llvm::LLVMContext> Ctx;
  std::unique_ptr<llvm::Module> Module;
  int LiftedCount = 0;
  int TotalCount = 0;
  // Source disassembly, populated only when HSA_HOTSWAP_DUMP_INPUT=1 for the
  // `.dis` debug dump; empty on the production path.
  std::string DisasmText;
  // Predicate-chain classifier observations that the cross-widening
  // analysis accepted (rather than refused) for this kernel. Surfaced
  // for diagnostic attribution; counters are zero on a clean lift.
  // TODO(naming): the `c5*` identifier is prototype-era jargon and
  // should be replaced with a domain-meaningful name before this lands.
  int C5SuppressedCount = 0;
  std::string C5SuppressionReason;
  bool UsesScratchPrivateSegment = false;
  uint32_t SourcePrivateSegmentFixedSize = 0;
  bool HasDivergentExec = false;
};

// Build a trapping stub for a kernel that cannot be translated. Keeps the
// translated code object loadable (so sibling kernels stay usable) while
// ensuring a dispatch of the stub fails loudly (llvm.trap / s_trap 2 on
// AMDGPU) rather than silently returning garbage. Preserves the source
// kernarg ABI via a byref placeholder so the descriptor's kernarg_segment_size
// matches `Meta.KernargSegmentSize`.
RaiseResult raiseStubKernel(const KernelMeta &Meta, llvm::StringRef KernelName);

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          llvm::StringRef CompilationTargetIsa = "",
          bool EnableWritelaneRewrite = true, bool EnableWaveNative = true);

llvm::Expected<RaiseResult>
raiseToIR(llvm::ArrayRef<uint8_t> TextBytes, llvm::StringRef SourceIsa,
          llvm::StringRef KernelName, const KernelMeta &Meta,
          uint64_t KernelOffset, uint64_t KernelSize,
          llvm::StringRef CompilationTargetIsa = "",
          bool EnableWritelaneRewrite = true, bool EnableWaveNative = true,
          bool AssumeHipGlobalOffsetZero = false,
          // Text-relative extents of all function symbols in the code object
          // (from listTextFunctionExtents). Lets the raiser follow a
          // call/branch into an outlined helper outside the selected kernel's
          // own extent and lift it alongside the caller. Empty (the default)
          // keeps the legacy behavior: any out-of-extent target is a
          // kernel-boundary violation.
          llvm::ArrayRef<KernelSymbolExtent> FunctionExtents = {});

} // namespace COMGR::hotswap

#endif
