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
// Each converted byte is checked against an oracle built on llvm::APFloat,
// which shares no code with the converters: APFloat does the format conversion
// and its round-half-to-even, and the oracle layers on only the three policy
// classes fp8-convert.h documents (NaN/Inf -> canonical target NaN, finite
// overflow -> saturate, FNUZ has no -0).  A hand-written oracle would have to
// restate the rounding rules the converters implement, and a subtle
// misunderstanding would then be baked into both sides; APFloat cannot drift
// with them.  Edge classes are additionally asserted explicitly: NaN, Inf
// (E5M2), subnormals, E4M3 OCP (240,448] saturation, and FNUZ->OCP
// round-half-to-even.

#include "hotswap/fp8-convert.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include "gtest/gtest.h"

#include <array>
#include <cstdint>

using namespace llvm;
using namespace COMGR::hotswap;

namespace {

// Run `Conv` over all 256 byte values at once and return the 256 folded
// results.  Builds a `<256 x i32>` constant [0, 1, ..., 255], calls the real
// emitter (which the ConstantFolder collapses to a constant), and extracts each
// lane.  Fails the test if any lane did not fold to a constant.
std::array<uint8_t, 256>
runConverter(llvm::function_ref<Value *(HotswapIRBuilder &, Value *)> Conv) {
  LLVMContext Ctx;
  Module M("fp8convtest", Ctx);
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
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

// --- APFloat-based oracle (shares no code with the converters) ---

// Convert one byte from \p From to \p To the way fp8-convert.h specifies.
// APFloat performs the format conversion and the round-half-to-even; only the
// documented policy classes are applied on top of it.
uint8_t oracleConvert(uint8_t Byte, const fltSemantics &From,
                      const fltSemantics &To) {
  const bool ToFnuz =
      &To == &APFloat::Float8E4M3FNUZ() || &To == &APFloat::Float8E5M2FNUZ();
  const uint8_t TargetNaN = ToFnuz ? 0x80 : 0x7F;

  APFloat V(From, APInt(8, Byte));
  // Neither target format has Inf, so Inf joins NaN in mapping to the target's
  // canonical NaN rather than saturating.
  if (V.isNaN() || V.isInfinity())
    return TargetNaN;

  const bool Neg = V.isNegative();
  bool LosesInfo = false;
  V.convert(To, APFloat::rmNearestTiesToEven, &LosesInfo);
  // A finite input that overflows the target comes back as NaN (no Inf to
  // round to); the converters saturate to the target max instead.
  if (V.isNaN() || V.isInfinity())
    V = APFloat::getLargest(To, Neg);
  if (ToFnuz && V.isZero())
    return 0x00; // FNUZ has no -0
  return static_cast<uint8_t>(V.bitcastToAPInt().getZExtValue());
}

// Run every byte through `Conv` and compare against the APFloat oracle.
// `std::hex` does not survive gtest's Message stream, so format explicitly --
// a converter bug reports bit patterns, and decimal would be unreadable.
void checkExhaustive(
    const char *Label,
    llvm::function_ref<Value *(HotswapIRBuilder &, Value *)> Conv,
    const fltSemantics &From, const fltSemantics &To) {
  auto Hex = [](unsigned V) {
    return "0x" + utohexstr(V, /*LowerCase=*/false, /*Width=*/2);
  };
  auto Got = runConverter(Conv);
  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    uint8_t Want = oracleConvert(static_cast<uint8_t>(Byte), From, To);
    EXPECT_EQ(Got[Byte], Want)
        << Label << " mismatch at byte " << Hex(Byte) << ": got "
        << Hex(Got[Byte]) << " want " << Hex(Want);
  }
}

} // namespace

TEST(Fp8Convert, OcpE4M3ToFnuzExhaustive) {
  checkExhaustive("OCP E4M3->FNUZ", convertOcpE4M3ToFnuz,
                  APFloat::Float8E4M3FN(), APFloat::Float8E4M3FNUZ());
}

TEST(Fp8Convert, OcpE5M2ToFnuzExhaustive) {
  checkExhaustive("OCP E5M2->FNUZ", convertOcpE5M2ToFnuz, APFloat::Float8E5M2(),
                  APFloat::Float8E5M2FNUZ());
}

TEST(Fp8Convert, FnuzE4M3ToOcpExhaustive) {
  checkExhaustive("FNUZ E4M3->OCP", convertFnuzE4M3ToOcp,
                  APFloat::Float8E4M3FNUZ(), APFloat::Float8E4M3FN());
}

TEST(Fp8Convert, FnuzE5M2ToOcpExhaustive) {
  checkExhaustive("FNUZ E5M2->OCP", convertFnuzE5M2ToOcp,
                  APFloat::Float8E5M2FNUZ(), APFloat::Float8E5M2());
}

// Decode every byte through the real emitter and compare against APFloat's
// own fp8 -> f32 conversion, which is exact (widening) and shares no code with
// the decoder.  NaN is compared by class, since the payload is unspecified.
void checkDecode(const char *Label, const fltSemantics &From, bool IsBf8,
                 bool IsFnuz) {
  LLVMContext Ctx;
  Module M("fp8dectest", Ctx);
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), false),
                                 GlobalValue::ExternalLinkage, "f", &M);
  HotswapIRBuilder B(BasicBlock::Create(Ctx, "entry", F));

  for (unsigned Byte = 0; Byte < 256; ++Byte) {
    Value *Out = decodeFp8ByteToF32(B, B.getInt32(Byte), IsBf8, IsFnuz);
    auto *CF = dyn_cast<ConstantFP>(Out);
    ASSERT_TRUE(CF != nullptr)
        << Label << " byte " << Byte << " did not constant-fold";
    APFloat Got = CF->getValueAPF();

    APFloat Want(From, APInt(8, Byte));
    bool LosesInfo = false;
    Want.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                 &LosesInfo);
    EXPECT_FALSE(LosesInfo) << Label << ": fp8 -> f32 must be exact";

    if (Want.isNaN()) {
      EXPECT_TRUE(Got.isNaN()) << Label << " byte 0x"
                               << utohexstr(Byte, false, 2) << " should be NaN";
      continue;
    }
    // bitwiseIsEqual distinguishes +0 from -0, which is the point for OCP.
    EXPECT_TRUE(Got.bitwiseIsEqual(Want))
        << Label << " byte 0x" << utohexstr(Byte, false, 2) << ": got "
        << Got.convertToFloat() << " want " << Want.convertToFloat();
  }
}

TEST(Fp8Convert, DecodeOcpE4M3ToF32Exhaustive) {
  checkDecode("OCP E4M3 decode", APFloat::Float8E4M3FN(), /*IsBf8=*/false,
              /*IsFnuz=*/false);
}

TEST(Fp8Convert, DecodeOcpE5M2ToF32Exhaustive) {
  checkDecode("OCP E5M2 decode", APFloat::Float8E5M2(), /*IsBf8=*/true,
              /*IsFnuz=*/false);
}

TEST(Fp8Convert, DecodeFnuzE4M3ToF32Exhaustive) {
  checkDecode("FNUZ E4M3 decode", APFloat::Float8E4M3FNUZ(), /*IsBf8=*/false,
              /*IsFnuz=*/true);
}

TEST(Fp8Convert, DecodeFnuzE5M2ToF32Exhaustive) {
  checkDecode("FNUZ E5M2 decode", APFloat::Float8E5M2FNUZ(), /*IsBf8=*/true,
              /*IsFnuz=*/true);
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

  // FNUZ has no Inf, so OCP E5M2 +/-Inf joins NaN at 0x80 rather than
  // saturating -- matching APFloat and the gfx942 f32->bf8 encode.
  EXPECT_EQ(E5ToFnuz[0x7C], 0x80u); // +Inf
  EXPECT_EQ(E5ToFnuz[0xFC], 0x80u); // -Inf
  EXPECT_EQ(E5ToFnuz[0x7D], 0x80u); // E5M2 NaN

  // FNUZ E4M3 NaN 0x80 -> OCP +0x7F.
  EXPECT_EQ(E4ToOcp[0x80], 0x7Fu);
  // FNUZ +0 -> OCP +0.
  EXPECT_EQ(E4ToOcp[0x00], 0x00u);
}
