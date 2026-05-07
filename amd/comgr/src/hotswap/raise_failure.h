//===- raise_failure.h - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_H
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap {

struct DecodedInst;

// Structured reason for a raise failure. Lives in its own header so the
// handler layer (`raise_context.h`) can depend on failure values
// without pulling in `RaiseResult` and the rest of the top-level
// `raiser.h` interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  // Caller-supplied input that the entry validator rejects before the
  // MC stack is even constructed. Today this fires on an empty or
  // non-AMDGPU `sourceISA` string, which would otherwise reach
  // `createMCDisassembler` and trip the `LLVM ERROR: disassembly not
  // yet supported for subtarget` `report_fatal_error` (process abort).
  // `detail` carries the offending input string.
  BadInput,
  UnsupportedOpcode,
};


struct RaiseFailure {
  RaiseFailureReason Reason = RaiseFailureReason::None;
  // Offending instruction mnemonic (e.g. `global_store_dwordx4`).
  std::string Mnemonic;
  // Encoding-format category (e.g. `VALU`, `FLAT`, `MUBUF`) — stable
  // bucketing key for the batch / corpus test summaries.
  std::string Format;
  // Byte offset inside the disassembled text section, in host order.
  // Zero for failures not tied to a specific instruction.
  uint64_t Offset = 0;
  // Optional human-readable context.
  std::string Detail;

  bool hasFailed() const { return Reason != RaiseFailureReason::None; }

  // Main loop: no handler claimed the CanonicalOp (either no TSFlags match
  // or every matching handler returned `handled=false` without
  // setting a more specific failure). `di` supplies the mnemonic /
  // offset; `format` is the human-readable encoding label.
  static RaiseFailure unsupportedOpcode(const DecodedInst &Di,
                                        llvm::StringRef Format);
};

} // namespace COMGR::hotswap

#endif
