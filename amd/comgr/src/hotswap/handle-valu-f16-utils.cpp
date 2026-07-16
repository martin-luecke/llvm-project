//===- handle-valu-f16-utils.cpp - F16 VALU helpers -----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-f16-utils.h"

#include "SIDefines.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace COMGR::hotswap {

void writeSelectedI16Bits(RaiseContext &Ctx, ParsedReg Dst, Value *Result,
                          bool DstHigh, StringRef MergeName) {
  writeSelectedI16Bits(Ctx, Dst, Result, DstHigh, MergeName, MergeName);
}

void writeSelectedI16Bits(RaiseContext &Ctx, ParsedReg Dst, Value *Result,
                          bool DstHigh, StringRef MergeLoName,
                          StringRef MergeHiName) {
  Value *Bits = Ctx.B.CreateZExt(Result, Ctx.I32Ty);
  Value *Old = Ctx.Regs.readReg32(Ctx.B, Dst);
  if (!DstHigh) {
    Value *High =
        Ctx.B.CreateAnd(Old, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u));
    Ctx.writeReg32(Dst, Ctx.B.CreateOr(High, Bits, MergeLoName));
    return;
  }

  Value *Low = Ctx.B.CreateAnd(Old, ConstantInt::get(Ctx.I32Ty, 0x0000FFFFu));
  Value *Shifted = Ctx.B.CreateShl(Bits, 16);
  Ctx.writeReg32(Dst, Ctx.B.CreateOr(Low, Shifted, MergeHiName));
}

} // namespace COMGR::hotswap
