//===- handle-valu-f16-utils.h - F16 VALU helpers -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared VOP3 F16 modifier helpers for VALU handlers.
//
// AMDGPU VOP3 F16 encodings carry source arithmetic modifiers and true16
// half-select bits in the decoded `srcN_modifiers` operands. For true16 forms,
// src0's modifier word also carries the destination-half selector; handlers
// must merge the 16-bit result into that half while preserving the other half
// of the destination VGPR.
//
// The helpers below provide both strict VOP3-only reads (modifier operand must
// be present) and e32/e64-shared reads (missing modifier operand means the
// default low-half, unmodified form). Unsupported modifier bits are refused
// rather than silently ignored.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H
#define HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H

#include "handle-valu-internal.h"

#include "llvm/ADT/StringRef.h"

namespace COMGR::hotswap {

// Merge a raw i16 result into the selected destination half using the same IR
// name for either half. The unselected half is preserved by reading the old
// destination VGPR value.
void writeSelectedI16Bits(RaiseContext &Ctx, ParsedReg Dst, llvm::Value *Result,
                          bool DstHigh, llvm::StringRef MergeName);
// Merge a raw i16 result into the selected destination half, using
// half-specific IR names for the low/high merge sites. The unselected half is
// preserved by reading the old destination VGPR value.
void writeSelectedI16Bits(RaiseContext &Ctx, ParsedReg Dst, llvm::Value *Result,
                          bool DstHigh, llvm::StringRef MergeLoName,
                          llvm::StringRef MergeHiName);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H
