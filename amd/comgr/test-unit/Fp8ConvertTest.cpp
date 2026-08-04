//===- Fp8ConvertTest.cpp - fp8 OCP<->FNUZ byte converter unit tests ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Unit tests for the per-byte-lane fp8/bf8 OCP<->FNUZ re-encoders in
// `fp8-convert.{h,cpp}`.  gfx940/941/942 (CDNA3) fp8 hardware reads bytes as
// FNUZ (E4M3FNUZ bias 8, E5M2FNUZ bias 16); gfx950/gfx12 use OCP (E4M3FN bias
// 7, E5M2 bias 15).  The raiser re-encodes at every gfx942 fp8 boundary, so a
// bit error in these converters silently corrupts every fp8 operand.
//
// The converters are IR emitters (`Value *convertX(HotswapIRBuilder&, Value*)`
// over a `<N x i32>` where each lane holds a byte).  HotswapIRBuilder carries a
// ConstantFolder, so a ConstantVector input folds the whole converter to a
// ConstantVector output with no target/codegen.  Each test feeds all 256 byte
// values as one `<256 x i32>` splat-free constant, runs the real emitter, and
// reads each folded lane back -- this exercises the shipping IR, not a C++
// mirror that could drift from it.
//
// Each converted byte is checked against an INDEPENDENT oracle derived from the
// format spec (decode the source byte to an exact rational, then re-encode in
// the target format with the documented rounding/saturation) -- deliberately
// not the converter's own bit-algebra, so the two must agree by construction,
// not by copy.  Edge classes are asserted explicitly: NaN, Inf (E5M2),
// subnormals, E4M3 OCP (240,448] saturation, and FNUZ->OCP round-half-to-even.

#include "hotswap/fp8-convert.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "gtest/gtest.h"

#include <array>
#include <cmath>
#include <cstdint>

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

// Run `Conv` over all 256 byte values at once and return the 256 folded
// results.  Builds a `<256 x i32>` constant [0, 1, ..., 255], calls the real
// emitter (which the ConstantFolder collapses to a constant), and extracts each
// lane.  Fails the test if any lane did not fold to a constant.
std::array<uint8_t, 256> runConverter(
    llvm::function_ref<Value *(HotswapIRBuilder &, Value *)> Conv) {
  LLVMContext Ctx;
  Module M("fp8convtest", Ctx);
  Function *F = Function::Create(
      FunctionType::get(Type::getVoidTy(Ctx), false),
      GlobalValue::ExternalLinkage, "f", &M);
  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", F);
  HotswapIRBuilder B(BB);

  Type *I32Ty = B.getInt32Ty();
  SmallVector<Constant *, 256> Lanes;
  for (unsigned I = 0; I < 256; ++I)
    Lanes.push_back(ConstantInt::get(I32Ty, I));
  Value *In = ConstantVector::get(Lanes);

  Value *Out = Conv(B, In);
  auto *OutC = dyn_cast<Constant>(Out);
  EXPECT_TRUE(OutC != nullptr) << "converter did not constant-fold";

  std::array<uint8_t, 256> Result{};
  for (unsigned I = 0; I < 256; ++I) {
    Constant *Lane = OutC ? OutC->getAggregateElement(I) : nullptr;
    auto *CI = dyn_cast_or_null<ConstantInt>(Lane);
    EXPECT_TRUE(CI != nullptr) << "lane " << I << " did not fold";
    Result[I] = CI ? static_cast<uint8_t>(CI->getZExtValue() & 0xFF) : 0;
  }
  return Result;
}

// --- Independent format oracles (spec-derived, not the converter algebra) ---

struct Fp8Field {
  int expBits;
  int mantBits;
  int bias;
};
constexpr Fp8Field OcpE4M3{4, 3, 7};
constexpr Fp8Field FnuzE4M3{4, 3, 8};
constexpr Fp8Field OcpE5M2{5, 2, 15};
constexpr Fp8Field FnuzE5M2{5, 2, 16};

// Decode a byte to (sign, numerator, denominator) rational, or flag
// NaN/Inf.  Value = (-1)^sign * num / den.  OCP has E4M3FN (no Inf, S.1111.111
// = NaN) and E5M2 (S.11111.00 = Inf, else NaN).  FNUZ: 0x80 is the sole NaN,
// no Inf, and 0x00 is the only zero (negative-zero encoding is NaN).
struct Decoded {
  bool isNaN = false;
  bool isInf = false;
  int sign = 0;
  // magnitude = num / den (den is a power of two)
  uint64_t num = 0;
  uint64_t den = 1;
};

Decoded decode(uint8_t Byte, const Fp8Field &F, bool Fnuz) {
  Decoded D;
  int sign = (Byte >> 7) & 1;
  int expMask = (1 << F.expBits) - 1;
  int mantMask = (1 << F.mantBits) - 1;
  int exp = (Byte >> F.mantBits) & expMask;
  int mant = Byte & mantMask;
  D.sign = sign;

  if (Fnuz) {
    if (Byte == 0x80) { D.isNaN = true; return D; }
    if (Byte == 0x00) { D.num = 0; D.den = 1; return D; }
  } else {
    if (F.expBits == 4) { // E4M3FN: no Inf
      if (exp == expMask && mant == mantMask) { D.isNaN = true; return D; }
    } else { // E5M2: Inf/NaN
      if (exp == expMask) {
        if (mant == 0) D.isInf = true; else D.isNaN = true;
        return D;
      }
    }
    if (Byte == 0x00 || Byte == 0x80) { D.num = 0; D.den = 1; return D; }
  }

  // value = 2^(e - bias) * (1 + mant/2^m)  [normal, exp!=0]
  //       = 2^(1 - bias) * (mant/2^m)      [subnormal, exp==0]
  int m = F.mantBits;
  uint64_t mantNum = (exp == 0) ? mant : ((1u << m) + mant);
  int e = (exp == 0) ? (1 - F.bias) : (exp - F.bias);
  // magnitude = mantNum / 2^m * 2^e
  int shift = e - m;
  if (shift >= 0) { D.num = mantNum << shift; D.den = 1; }
  else { D.num = mantNum; D.den = 1ull << (-shift); }
  return D;
}

// Encode a rational magnitude (num/den, den a power of two) into the target
// format with round-half-to-even, returning the byte (sign applied by caller).
// Saturates to max-finite on overflow.  This is the spec oracle; it is
// intentionally structured differently from the converter's byte algebra.
uint8_t encodeMagnitude(uint64_t num, uint64_t den, int sign,
                        const Fp8Field &F, bool Fnuz) {
  int expMask = (1 << F.expBits) - 1;
  int mantMask = (1 << F.mantBits) - 1;
  int m = F.mantBits;
  uint8_t signBit = static_cast<uint8_t>(sign << 7);

  if (num == 0)
    return Fnuz ? 0x00 : signBit; // FNUZ has only +0

  // Normalize num/den to 1.xxx * 2^exp2 (or subnormal).  Find exp2 with
  // 2^exp2 <= num/den < 2^(exp2+1).
  // Work in a scaled integer domain: value = num/den.
  // Compute floor(log2(num/den)).
  int exp2 = 0;
  // Bring value into [1,2) by scaling num/den.
  long double v = static_cast<long double>(num) / static_cast<long double>(den);
  while (v >= 2.0L) { v /= 2.0L; ++exp2; }
  while (v < 1.0L)  { v *= 2.0L; --exp2; }

  int storedExp = exp2 + F.bias;
  int maxStoredExp = Fnuz ? expMask : (F.expBits == 4 ? expMask : expMask - 1);
  // For OCP E4M3FN max finite is exp==15,mant==6 (mant==7 is NaN); handled by
  // saturation below.  For E5M2 OCP max finite exp==30.
  if (storedExp >= (Fnuz ? (expMask + 1) : (F.expBits == 4 ? expMask + 1
                                                          : expMask))) {
    // Overflow -> saturate to max finite of the target.
    if (Fnuz) return static_cast<uint8_t>(signBit | 0x7F);
    if (F.expBits == 4) return static_cast<uint8_t>(signBit | 0x7E); // 240 OCP
    return static_cast<uint8_t>(signBit | ((expMask - 1) << m) | mantMask);
  }

  if (storedExp <= 0) {
    // Subnormal: value = mant/2^m * 2^(1-bias).  mant = round(v * 2^exp2 /
    // 2^(1-bias) * 2^m) but simplest: scale original.
    long double sub = (static_cast<long double>(num) /
                       static_cast<long double>(den)) /
                      std::pow(2.0L, 1 - F.bias) * std::pow(2.0L, m);
    long double flo = std::floor(sub);
    long double frac = sub - flo;
    uint64_t mant = static_cast<uint64_t>(flo);
    if (frac > 0.5L) mant++;
    else if (frac == 0.5L) mant += (mant & 1); // round-half-to-even
    if (mant == 0) return Fnuz ? 0x00 : signBit;
    if (mant > (uint64_t)mantMask) {
      // rounded up into the smallest normal
      return static_cast<uint8_t>(signBit | (1 << m));
    }
    return static_cast<uint8_t>(signBit | mant);
  }

  // Normal: mantissa = round((v - 1) * 2^m), RNE.
  long double frac = (v - 1.0L) * static_cast<long double>(1u << m);
  long double flo = std::floor(frac);
  long double f = frac - flo;
  uint64_t mant = static_cast<uint64_t>(flo);
  if (f > 0.5L) mant++;
  else if (f == 0.5L) mant += (mant & 1);
  if (mant > (uint64_t)mantMask) { mant = 0; storedExp++; }
  if (storedExp > maxStoredExp) {
    if (Fnuz) return static_cast<uint8_t>(signBit | 0x7F);
    if (F.expBits == 4) return static_cast<uint8_t>(signBit | 0x7E);
    return static_cast<uint8_t>(signBit | ((expMask - 1) << m) | mantMask);
  }
  return static_cast<uint8_t>(signBit | (storedExp << m) | mant);
}

// OCP -> FNUZ oracle for one byte.
uint8_t oracleOcpToFnuz(uint8_t Byte, const Fp8Field &Ocp,
                        const Fp8Field &Fnuz) {
  Decoded D = decode(Byte, Ocp, /*Fnuz=*/false);
  if (D.isNaN || D.isInf) {
    // OCP NaN and E5M2 Inf both map to FNUZ; Inf saturates to max finite,
    // NaN -> 0x80.
    if (D.isInf) return static_cast<uint8_t>((D.sign << 7) | 0x7F);
    return 0x80;
  }
  return encodeMagnitude(D.num, D.den, D.sign, Fnuz, /*Fnuz=*/true);
}

// FNUZ -> OCP oracle for one byte.
uint8_t oracleFnuzToOcp(uint8_t Byte, const Fp8Field &Fnuz,
                        const Fp8Field &Ocp) {
  Decoded D = decode(Byte, Fnuz, /*Fnuz=*/true);
  if (D.isNaN)
    // FNUZ has a single NaN encoding (0x80); the converter maps it to the
    // sign-less canonical OCP +NaN 0x7F, not a sign-preserving NaN.
    return 0x7F;
  return encodeMagnitude(D.num, D.den, D.sign, Ocp, /*Fnuz=*/false);
}

} // namespace

TEST(Fp8Convert, OcpE4M3ToFnuzExhaustive) {
  auto Got = runConverter(convertOcpE4M3ToFnuz);
  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    uint8_t Want = oracleOcpToFnuz(static_cast<uint8_t>(Byte), OcpE4M3, FnuzE4M3);
    EXPECT_EQ(Got[Byte], Want)
        << "OCP E4M3->FNUZ mismatch at byte 0x" << std::hex << Byte
        << " got 0x" << (unsigned)Got[Byte] << " want 0x" << (unsigned)Want;
  }
}

TEST(Fp8Convert, OcpE5M2ToFnuzExhaustive) {
  auto Got = runConverter(convertOcpE5M2ToFnuz);
  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    uint8_t Want = oracleOcpToFnuz(static_cast<uint8_t>(Byte), OcpE5M2, FnuzE5M2);
    EXPECT_EQ(Got[Byte], Want)
        << "OCP E5M2->FNUZ mismatch at byte 0x" << std::hex << Byte
        << " got 0x" << (unsigned)Got[Byte] << " want 0x" << (unsigned)Want;
  }
}

TEST(Fp8Convert, FnuzE4M3ToOcpExhaustive) {
  auto Got = runConverter(convertFnuzE4M3ToOcp);
  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    uint8_t Want = oracleFnuzToOcp(static_cast<uint8_t>(Byte), FnuzE4M3, OcpE4M3);
    EXPECT_EQ(Got[Byte], Want)
        << "FNUZ E4M3->OCP mismatch at byte 0x" << std::hex << Byte
        << " got 0x" << (unsigned)Got[Byte] << " want 0x" << (unsigned)Want;
  }
}

TEST(Fp8Convert, FnuzE5M2ToOcpExhaustive) {
  auto Got = runConverter(convertFnuzE5M2ToOcp);
  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    uint8_t Want = oracleFnuzToOcp(static_cast<uint8_t>(Byte), FnuzE5M2, OcpE5M2);
    EXPECT_EQ(Got[Byte], Want)
        << "FNUZ E5M2->OCP mismatch at byte 0x" << std::hex << Byte
        << " got 0x" << (unsigned)Got[Byte] << " want 0x" << (unsigned)Want;
  }
}

// Explicit edge-class pins (independent of the oracle loop above) so a
// regression names the exact special case it broke.
TEST(Fp8Convert, EdgeClasses) {
  auto E4ToFnuz = runConverter(convertOcpE4M3ToFnuz);
  auto E5ToFnuz = runConverter(convertOcpE5M2ToFnuz);
  auto E4ToOcp = runConverter(convertFnuzE4M3ToOcp);

  // OCP E4M3 NaN (S.1111.111) -> FNUZ NaN 0x80.
  EXPECT_EQ(E4ToFnuz[0x7F], 0x80u);
  EXPECT_EQ(E4ToFnuz[0xFF], 0x80u);
  // OCP E4M3 (240,448] saturates to FNUZ max finite 240 (0x7F), sign kept.
  EXPECT_EQ(E4ToFnuz[0x78], 0x7Fu); // exp=15,mant=0 -> saturate
  EXPECT_EQ(E4ToFnuz[0xF8], 0xFFu); // negative saturate
  // +/-0 both map to FNUZ +0.
  EXPECT_EQ(E4ToFnuz[0x00], 0x00u);
  EXPECT_EQ(E4ToFnuz[0x80], 0x00u);

  // OCP E5M2 +Inf (0x7C) saturates to FNUZ max finite 0x7F; NaN -> 0x80.
  EXPECT_EQ(E5ToFnuz[0x7C], 0x7Fu);
  EXPECT_EQ(E5ToFnuz[0xFC], 0xFFu);
  EXPECT_EQ(E5ToFnuz[0x7D], 0x80u); // E5M2 NaN

  // FNUZ E4M3 NaN 0x80 -> OCP +0x7F.
  EXPECT_EQ(E4ToOcp[0x80], 0x7Fu);
  // FNUZ +0 -> OCP +0.
  EXPECT_EQ(E4ToOcp[0x00], 0x00u);
}