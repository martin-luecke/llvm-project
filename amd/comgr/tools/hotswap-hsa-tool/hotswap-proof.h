//===- hotswap-proof.h - HotSwap proof log support --------------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TOOLS_HOTSWAP_HSA_TOOL_PROOF_H
#define COMGR_TOOLS_HOTSWAP_HSA_TOOL_PROOF_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap::hsa_tool {

std::string jsonEscape(llvm::StringRef Value);

struct ProofLine {
  llvm::StringRef Path;
  llvm::StringRef Fields;
  uint64_t ProcessId = 0;
};

bool appendProofLine(const ProofLine &Line);

} // namespace COMGR::hotswap::hsa_tool

#endif
