//===- fp8-convert.cpp - Hotswap transpiler ------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fp8-convert.h"
#include "isa-profile.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"

using namespace llvm;

namespace COMGR::hotswap {

Fp8Format fp8FormatOf(const ISAProfile &P) {
  if (!(P.HasFP8Insts || P.HasFP8ConversionInsts))
    return Fp8Format::None;
  if (P.HasGfx950Insts)
    return Fp8Format::OCP;
  if (P.HasMfma)
    return Fp8Format::FNUZ;
  return Fp8Format::OCP;
}

std::optional<bool> fp8Reencode(const ISAProfile &Src, const ISAProfile &Tgt,
                                Fp8Dir Dir) {
  Fp8Format S = fp8FormatOf(Src), T = fp8FormatOf(Tgt);
  if (S == Fp8Format::None || T == Fp8Format::None || S == T)
    return std::nullopt;
  return (Dir == Fp8Dir::SrcToTgt ? T : S) == Fp8Format::FNUZ;
}

namespace {

// Helper to build an N-lane i32 splat constant matching `Bytes`'s vector type.
struct ByteVecHelper {
  HotswapIRBuilder &B;
  unsigned N;
  IntegerType *I32Ty;
  ByteVecHelper(HotswapIRBuilder &B, Value *Bytes) : B(B) {
    auto *VecTy = cast<FixedVectorType>(Bytes->getType());
    N = VecTy->getNumElements();
    I32Ty = cast<IntegerType>(VecTy->getElementType());
  }
  Constant *splat(uint64_t V) const {
    return ConstantVector::getSplat(ElementCount::getFixed(N),
                                    ConstantInt::get(I32Ty, V));
  }
};

} // namespace

// OCP E4M3FN (bias 7) -> FNUZ E4M3 (bias 8).
//   normals: stored exponent +1 (mantissa identical, no rounding);
//   subnormals: byte = sign | (mant << 1) (exact); +/-0 -> +0;
//   exp==15: mant==7 -> 0x80 (NaN); else saturate to sign|0x7F (+/-240).
Value *convertOcpE4M3ToFnuz(HotswapIRBuilder &B, Value *Bytes) {
  ByteVecHelper H(B, Bytes);
  auto S = [&](uint64_t V) { return H.splat(V); };
  Value *Sign = B.CreateShl(B.CreateAnd(B.CreateLShr(Bytes, S(7)), S(1)), S(7));
  Value *Exp = B.CreateAnd(B.CreateLShr(Bytes, S(3)), S(0xF));
  Value *Mant = B.CreateAnd(Bytes, S(0x7));
  Value *Norm = B.CreateOr(
      Sign, B.CreateOr(B.CreateShl(B.CreateAdd(Exp, S(1)), S(3)), Mant));
  Value *Sub = B.CreateSelect(B.CreateICmpEQ(Mant, S(0)), S(0),
                              B.CreateOr(Sign, B.CreateShl(Mant, S(1))));
  Value *Top = B.CreateSelect(B.CreateICmpEQ(Mant, S(7)), S(0x80),
                              B.CreateOr(Sign, S(0x7F)));
  Value *R = B.CreateSelect(B.CreateICmpEQ(Exp, S(0xF)), Top, Norm);
  return B.CreateSelect(B.CreateICmpEQ(Exp, S(0)), Sub, R, "e4m3_fnuz");
}

// OCP E5M2 (bias 15) -> FNUZ E5M2 (bias 16). Same shape as E4M3 but 5-bit exp,
// 2-bit mantissa; exp==31: Inf (mant==0) saturates, NaN -> 0x80.
Value *convertOcpE5M2ToFnuz(HotswapIRBuilder &B, Value *Bytes) {
  ByteVecHelper H(B, Bytes);
  auto S = [&](uint64_t V) { return H.splat(V); };
  Value *Sign = B.CreateShl(B.CreateAnd(B.CreateLShr(Bytes, S(7)), S(1)), S(7));
  Value *Exp = B.CreateAnd(B.CreateLShr(Bytes, S(2)), S(0x1F));
  Value *Mant = B.CreateAnd(Bytes, S(0x3));
  Value *Norm = B.CreateOr(
      Sign, B.CreateOr(B.CreateShl(B.CreateAdd(Exp, S(1)), S(2)), Mant));
  Value *Sub = B.CreateSelect(B.CreateICmpEQ(Mant, S(0)), S(0),
                              B.CreateOr(Sign, B.CreateShl(Mant, S(1))));
  Value *Top = B.CreateSelect(B.CreateICmpEQ(Mant, S(0)),
                              B.CreateOr(Sign, S(0x7F)), S(0x80));
  Value *R = B.CreateSelect(B.CreateICmpEQ(Exp, S(0x1F)), Top, Norm);
  return B.CreateSelect(B.CreateICmpEQ(Exp, S(0)), Sub, R, "e5m2_fnuz");
}

// FNUZ E4M3 (bias 8) -> OCP E4M3FN (bias 7).
//   normals (exp>=2): stored exponent -1 (exact);
//   exp<=1: OCP subnormal, mant' = round-half-to-even(N/2) with
//           N = (exp==1 ? 8 : 0) + mant; 0x80 (NaN) -> 0x7F.
// FNUZ range is a subset of OCP except its finer subnormals, which round.
Value *convertFnuzE4M3ToOcp(HotswapIRBuilder &B, Value *Bytes) {
  ByteVecHelper H(B, Bytes);
  auto S = [&](uint64_t V) { return H.splat(V); };
  Value *Sign = B.CreateShl(B.CreateAnd(B.CreateLShr(Bytes, S(7)), S(1)), S(7));
  Value *Exp = B.CreateAnd(B.CreateLShr(Bytes, S(3)), S(0xF));
  Value *Mant = B.CreateAnd(Bytes, S(0x7));
  Value *Norm = B.CreateOr(
      Sign, B.CreateOr(B.CreateShl(B.CreateSub(Exp, S(1)), S(3)), Mant));
  Value *NVal =
      B.CreateAdd(B.CreateSelect(B.CreateICmpEQ(Exp, S(1)), S(8), S(0)), Mant);
  Value *Rne =
      B.CreateAdd(B.CreateLShr(NVal, S(1)),
                  B.CreateSelect(B.CreateICmpEQ(B.CreateAnd(NVal, S(3)), S(3)),
                                 S(1), S(0)));
  Value *Sub = B.CreateOr(Sign, Rne);
  Value *R = B.CreateSelect(B.CreateICmpUGE(Exp, S(2)), Norm, Sub);
  return B.CreateSelect(B.CreateICmpEQ(Bytes, S(0x80)), S(0x7F), R, "e4m3_ocp");
}

// FNUZ E5M2 (bias 16) -> OCP E5M2 (bias 15). Same structure as E4M3 with 5-bit
// exp / 2-bit mantissa.
Value *convertFnuzE5M2ToOcp(HotswapIRBuilder &B, Value *Bytes) {
  ByteVecHelper H(B, Bytes);
  auto S = [&](uint64_t V) { return H.splat(V); };
  Value *Sign = B.CreateShl(B.CreateAnd(B.CreateLShr(Bytes, S(7)), S(1)), S(7));
  Value *Exp = B.CreateAnd(B.CreateLShr(Bytes, S(2)), S(0x1F));
  Value *Mant = B.CreateAnd(Bytes, S(0x3));
  Value *Norm = B.CreateOr(
      Sign, B.CreateOr(B.CreateShl(B.CreateSub(Exp, S(1)), S(2)), Mant));
  Value *NVal =
      B.CreateAdd(B.CreateSelect(B.CreateICmpEQ(Exp, S(1)), S(4), S(0)), Mant);
  Value *Rne =
      B.CreateAdd(B.CreateLShr(NVal, S(1)),
                  B.CreateSelect(B.CreateICmpEQ(B.CreateAnd(NVal, S(3)), S(3)),
                                 S(1), S(0)));
  Value *Sub = B.CreateOr(Sign, Rne);
  Value *R = B.CreateSelect(B.CreateICmpUGE(Exp, S(2)), Norm, Sub);
  return B.CreateSelect(B.CreateICmpEQ(Bytes, S(0x80)), S(0x7F), R, "e5m2_ocp");
}

Value *convertFp8Dword(HotswapIRBuilder &B, Value *Dword, bool IsBf8,
                       bool ToFnuz) {
  auto *I32Ty = B.getInt32Ty();
  auto *Vec4I8 = FixedVectorType::get(B.getInt8Ty(), 4);
  Constant *Shifts = ConstantVector::get(
      {ConstantInt::get(I32Ty, 0), ConstantInt::get(I32Ty, 8),
       ConstantInt::get(I32Ty, 16), ConstantInt::get(I32Ty, 24)});
  Constant *ByteMask = ConstantVector::getSplat(ElementCount::getFixed(4),
                                                ConstantInt::get(I32Ty, 0xFF));
  Value *Splat = B.CreateVectorSplat(4, Dword, "fp8_splat");
  Value *Bytes =
      B.CreateAnd(B.CreateLShr(Splat, Shifts), ByteMask, "fp8_bytes");
  Value *Conv = ToFnuz ? (IsBf8 ? convertOcpE5M2ToFnuz(B, Bytes)
                                : convertOcpE4M3ToFnuz(B, Bytes))
                       : (IsBf8 ? convertFnuzE5M2ToOcp(B, Bytes)
                                : convertFnuzE4M3ToOcp(B, Bytes));
  Value *ConvBytes = B.CreateTrunc(Conv, Vec4I8, "fp8_conv_bytes");
  return B.CreateBitCast(ConvBytes, I32Ty, "fp8_conv_dw");
}

void convertFp8DwordsInPlace(HotswapIRBuilder &B,
                             SmallVectorImpl<Value *> &Dwords, bool IsBf8,
                             bool ToFnuz) {
  for (Value *&Dw : Dwords)
    Dw = convertFp8Dword(B, Dw, IsBf8, ToFnuz);
}

} // namespace COMGR::hotswap
