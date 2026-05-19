//===- handle-valu-f16-utils.h - F16 VALU helpers -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H
#define HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H

#include "handle-valu-internal.h"

#include "llvm/ADT/StringRef.h"

namespace COMGR::hotswap {

bool readRequiredVOP3F16SrcMods(const DecodedInst &Di, HandlerResult &Hr,
                                unsigned SrcIndex, llvm::StringRef OpName,
                                unsigned &Mods);

bool readOptionalVOP3F16SrcMods(const DecodedInst &Di, HandlerResult &Hr,
                                unsigned SrcIndex, llvm::StringRef OpName,
                                unsigned &Mods);

llvm::Value *readOpSelF16(RaiseContext &Ctx, const DecodedInst &Di,
                          OpResolver &Op, HandlerResult &Hr,
                          unsigned SrcIndex, llvm::StringRef OpName);

llvm::Value *readOptionalOpSelF16(RaiseContext &Ctx, const DecodedInst &Di,
                                  OpResolver &Op, HandlerResult &Hr,
                                  unsigned SrcIndex, llvm::StringRef OpName);

bool readVOP3F16DstHigh(const DecodedInst &Di, HandlerResult &Hr,
                        llvm::StringRef OpName, bool &DstHigh);

void writeOpSelF16(RaiseContext &Ctx, OpResolver &Op, llvm::Value *Result,
                   bool DstHigh, llvm::StringRef MergeLoName = "f16_merge_lo",
                   llvm::StringRef MergeHiName = "f16_merge_hi");

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_HANDLE_VALU_F16_UTILS_H
