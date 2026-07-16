//===- handle-valu-vop3p.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"

#include "canonical-op.h"
#include "wmma-lowering.h"

#include "SIDefines.h"            // SISrcMods::NEG
#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::getNamedOperandIdx, AMDGPU::OpName
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

#include <climits>

using namespace llvm;

namespace COMGR::hotswap {

namespace {

struct PackedSrcOptions {
  // Register operands in the packed-f32 family are VGPR pairs that should be
  // read as `<2 x elem>` directly. Packed-f16/i16 operands are one i32 VGPR
  // whose low/high halves are bitcast to `<2 x elem>`.
  bool RegisterSourceIsVector = false;
  // Packed-f32 immediates are scalar 32-bit literals broadcast to both lanes.
  // Packed-f16/i16 immediates are raw packed i32 payloads decoded by LLVM MC.
  bool ImmediateIsScalarBroadcast = false;
  // Floating-point packed families use NEG / NEG_HI as per-lane fneg bits.
  // Integer packed families reject those bits before calling the helper.
  bool ApplyFloatNeg = false;
  // IRBuilder base name used for temporary values from this source family.
  const char *Name = "pk_src";
};

StringRef diagnosticMnemonic(const DecodedInst &Di) {
  return Di.Mnemonic.empty() ? StringRef(canonicalOpName(Di.CanonOp))
                             : StringRef(Di.Mnemonic);
}

Error readSourceMods(const DecodedInst &Di, OpResolver &Op, unsigned NumSrcs,
                     unsigned AllowedMods, unsigned Mods[3]) {
  StringRef InstrName = diagnosticMnemonic(Di);
  if (Op.nSrcs() < NumSrcs)
    return RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3P", InstrName + " requires more source operands");

  for (unsigned I = 0; I < NumSrcs; ++I) {
    unsigned ModIdx = Di.ModMap[I];
    if (ModIdx == UINT_MAX || !Di.isImm(ModIdx))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", InstrName + " missing immediate srcN_modifiers operand");

    Mods[I] = static_cast<unsigned>(Di.getImm(ModIdx));
    if ((Mods[I] & ~AllowedMods) != 0)
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", InstrName + " has unsupported srcN_modifiers bits");
  }
  return Error::success();
}

Error readPackedSrcMods(const DecodedInst &Di, OpResolver &Op, unsigned NumSrcs,
                        unsigned AllowedMods, unsigned Mods[3]) {
  if (Error Err = readSourceMods(Di, Op, NumSrcs, AllowedMods, Mods))
    return Err;

  StringRef InstrName = diagnosticMnemonic(Di);
  for (unsigned I = 0; I < NumSrcs; ++I) {
    unsigned SrcIdx = Op.srcIdx(I);
    if (!Di.isReg(SrcIdx) && !Di.isImm(SrcIdx))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          InstrName + " source is neither a register nor an immediate");
  }
  return Error::success();
}

Value *readPacked2Src(RaiseContext &Ctx, OpResolver &Op, unsigned I,
                      Type *ElemTy, unsigned Mods,
                      const PackedSrcOptions &Opts) {
  auto *VecTy = FixedVectorType::get(ElemTy, 2);
  Value *NatLo = nullptr;
  Value *NatHi = nullptr;

  if (Opts.RegisterSourceIsVector && Op.isSrcReg(I)) {
    Value *Vec = Ctx.Regs.readRegVec(Ctx.B, Op.srcReg(I), VecTy);
    NatLo = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(0));
    NatHi = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(1));
  } else if (Opts.ImmediateIsScalarBroadcast && !Op.isSrcReg(I)) {
    Value *Scalar = Ctx.B.CreateBitCast(Op.src(I), ElemTy);
    NatLo = Scalar;
    NatHi = Scalar;
  } else {
    Value *Raw = Op.src(I);
    if (Raw->getType() != Ctx.I32Ty)
      Raw = Ctx.B.CreateBitCast(Raw, Ctx.I32Ty);
    Value *Vec = Ctx.B.CreateBitCast(Raw, VecTy, Opts.Name);
    NatLo = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(0));
    NatHi = Ctx.B.CreateExtractElement(Vec, static_cast<uint64_t>(1));
  }

  Value *Lo = (Mods & SISrcMods::OP_SEL_0) ? NatHi : NatLo;
  Value *Hi = (Mods & SISrcMods::OP_SEL_1) ? NatHi : NatLo;

  if (Opts.ApplyFloatNeg) {
    if (Mods & SISrcMods::NEG)
      Lo = Ctx.B.CreateFNeg(Lo, (Twine(Opts.Name) + "_neg_lo").str());
    if (Mods & SISrcMods::NEG_HI)
      Hi = Ctx.B.CreateFNeg(Hi, (Twine(Opts.Name) + "_neg_hi").str());
  }

  Value *R = UndefValue::get(VecTy);
  R = Ctx.B.CreateInsertElement(R, Lo, static_cast<uint64_t>(0));
  R = Ctx.B.CreateInsertElement(R, Hi, static_cast<uint64_t>(1));
  return R;
}

Value *applyF32InputMods(RaiseContext &Ctx, Value *V, unsigned Mods,
                         const Twine &Name) {
  if (V->getType() != Ctx.F32Ty)
    V = Ctx.B.CreateBitCast(V, Ctx.F32Ty);
  if (Mods & SISrcMods::ABS)
    V = Ctx.B.CreateUnaryIntrinsic(Intrinsic::fabs, V, nullptr,
                                   (Name + "_abs").str());
  if (Mods & SISrcMods::NEG)
    V = Ctx.B.CreateFNeg(V, (Name + "_neg").str());
  return V;
}

Value *readMixF32Src(RaiseContext &Ctx, OpResolver &Op, unsigned I,
                     Type *NarrowTy, unsigned Mods, StringRef CvtName) {
  Value *Raw = Op.src(I);
  if ((Mods & SISrcMods::OP_SEL_1) == 0)
    return applyF32InputMods(Ctx, Raw, Mods, "mix_full");

  if (Raw->getType() == Ctx.F32Ty)
    Raw = Ctx.B.CreateBitCast(Raw, Ctx.I32Ty);

  Value *Bits = nullptr;
  bool IsImmediateOperand = !Op.isSrcReg(I);
  if (!IsImmediateOperand && (Mods & SISrcMods::OP_SEL_0))
    Bits =
        Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Raw, 16), Type::getInt16Ty(Ctx.C));
  else
    Bits = Ctx.B.CreateTrunc(Raw, Type::getInt16Ty(Ctx.C));

  Value *NarrowVal = Ctx.B.CreateBitCast(Bits, NarrowTy);
  Value *Extended = Ctx.B.CreateFPExt(NarrowVal, Ctx.F32Ty, CvtName);
  return applyF32InputMods(Ctx, Extended, Mods, CvtName);
}

// Read the C (accumulator) operand of a WMMA instruction, handling the
// three encoding shapes LLVM's AMDGPU backend emits:
//
//   * _twoaddr form: C is tied to D (same VGPR slot, no separate `src2`
//     operand on the disassembled line). `op.isSrcReg(2)` is TRUE and
//     `srcReg(2)` returns the D VGPR -- we read the live VGPR value.
//   * _threeaddr form with a VGPR C: `isSrcReg(2)` TRUE and `srcReg(2)`
//     returns the explicit C VGPR. Same path as twoaddr -- just a
//     different VGPR index.
//   * _threeaddr form with an inline-constant C: LLVM picks this
//     encoding whenever the accumulator source is a constant that fits
//     in the VOP3P src2 inline-constant table (the important case is
//     `C = 0`, which Clang emits for every fresh accumulator built from
//     a zero-initialised `v8f c = {0, ..., 0}`). Here `isSrcReg(2)` is
//     FALSE; we MUST materialise the inline constant directly.
//
// The previous fallback `srcC = dest` was silently wrong for the third
// case: reading the D VGPR before the WMMA writes to it surfaces
// whatever stale (or undef) bits happened to be in those 8 VGPR slots,
// which on a cold kernel is typically zero by accident for the first
// WMMA in a wave but nondeterministic for any subsequent WMMA whose
// D range was never explicitly zero-initialised by the SGPR/VGPR
// prologue. In the `wmma_parallel{2,4,16}` probes the second and
// later WMMAs land on fresh D VGPRs (v[24:31], v[32:39], ...) that
// the compiler skipped zeroing -- precisely because it knew the
// threeaddr-imm-0 encoding would satisfy C.
//
// We handle only inline constant `0` today: it is the only src2 inline
// the AMDGPU backend actually emits for the WMMA family (Clang folds
// non-zero accumulator constants through a VGPR mov before the WMMA).
// Any other immediate surfaces as a structured `unsupportedInstructionForm`
// failure rather than silently miscompiling.
//
// On failure the helper returns an Error; the caller must short-circuit.
Expected<Value *> readWMMAAccumC(RaiseContext &Ctx, const DecodedInst &Di,
                                 OpResolver &Op, const ParsedReg &Dest,
                                 llvm::Type *CdIrTy) {
  if (Op.nSrcs() < 3) {
    // No src2 operand on the instruction at all (e.g. a hypothetical
    // encoding with C implicitly zero and no disassembler-surfaced
    // slot). Safest to refuse -- the caller expects to have read C.
    return RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3P",
        "WMMA instruction has no src2 (accumulator) operand; "
        "cannot recover C input");
  }
  if (Op.isSrcReg(2)) {
    ParsedReg SrcC = Op.srcReg(2);
    return Ctx.Regs.readRegVec(Ctx.B, SrcC, CdIrTy);
  }
  // Inline-constant src2. Today we only model `0`.
  unsigned SrcIdx2 = Op.srcIdx(2);
  if (!Di.isImm(SrcIdx2)) {
    // Could be a symbolic constant slot (e.g. SRC_EXEC_LO/HI, SRC_PC).
    // None of those are valid semantics for a WMMA accumulator; refuse.
    return RaiseFailure::unsupportedInstructionForm(
        Di, "VOP3P",
        "WMMA src2 is neither a register nor an immediate; no "
        "accumulator C input path is defined for this encoding");
  }
  int64_t ImmC = Di.getImm(SrcIdx2);
  if (ImmC == 0)
    return llvm::ConstantAggregateZero::get(CdIrTy);
  (void)Dest;
  return RaiseFailure::unsupportedInstructionForm(
      Di, "VOP3P",
      "WMMA src2 inline-constant other than 0 is not yet modelled; "
      "extend readWMMAAccumC if a corpus kernel surfaces this");
}

} // namespace

Value *raiseCndmaskWaveCondition(RaiseContext &Ctx, const DecodedInst &Di,
                                 OpResolver &Op) {
  Value *Cond = nullptr;
  if (Op.nSrcs() >= 3 && Di.isReg(Op.srcIdx(2))) {
    ParsedReg CondReg = Ctx.parseReg(Di.getReg(Op.srcIdx(2)), Op.srcIdx(2));
    if (CondReg.RegKind == ParsedReg::SGPR) {
      // Preferred path: a V_CMP_*_e64 in the current BB wrote this
      // SGPR and no intervening scalar write has invalidated the
      // cached per-lane `i1`. Use the `i1` directly -- it carries
      // the full target-hardware ballot without the cross-widening
      // narrow-write information loss (the SGPR itself holds only
      // the source-width-truncated 32-bit projection). See
      // hotswap/docs/sgpr-wave-mask-translation.md section 3.1 for
      // the full contract and
      // `RaiseContext::lastSgprWaveMaskI1` for the invariants that
      // make this lookup sound.
      if (Value *FreshCmp = Ctx.lookupSgprWaveMaskI1(CondReg.BaseIdx)) {
        Cond = FreshCmp;
      } else {
        // Fallback: no fresh V_CMP writer in this BB (or the cache
        // was invalidated by a scalar SGPR write, or we crossed a
        // BB boundary). Route through the projection's per-lane
        // extractor, mirroring `readVCCAsWaveMask`'s consumer
        // symmetry. This path is correct for same-wave and
        // modulo-replication same-width cases, and lossy only in
        // the documented wave32 -> wave64 cross-widening narrow-
        // write case (where recovering the upper-half lanes'
        // compare results is impossible from the 32-bit SGPR --
        // those bits were destroyed at the writer's truncate).
        Value *CondVal = Ctx.Isa.isWave32()
                             ? Ctx.Regs.loadSGPR32(Ctx.B, CondReg.BaseIdx)
                             : Ctx.Regs.loadSGPR64(Ctx.B, CondReg.BaseIdx);
        Value *Fallback =
            Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, CondVal);
        // Cross-BB path: prefer the memory-backed shadow if valid.
        // This avoids carrying non-dominating `i1` SSA values across
        // blocks while still preserving the full EXEC-width compare mask.
        if (Value *ShadowValid = Ctx.loadSgprWaveMaskValid(CondReg.BaseIdx)) {
          Value *ShadowExec = Ctx.loadSgprWaveMaskExec(CondReg.BaseIdx);
          Value *ShadowI1 =
              Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, ShadowExec);
          Cond = Ctx.B.CreateSelect(ShadowValid, ShadowI1, Fallback,
                                    "sgpr_mask_shadow_sel");
        } else {
          Cond = Fallback;
        }
      }
    } else if (CondReg.RegKind == ParsedReg::VCC_HI_SCRATCH ||
               CondReg.RegKind == ParsedReg::EXEC_HI_SCRATCH) {
      // Wave32-source vcc_hi / exec_hi are free general-purpose scalars
      // (see ParsedReg::VCC_HI_SCRATCH). Read the scratch slot and project
      // per-lane, not loadVCC (which would read the real VCC).
      Value *CondVal = Ctx.Regs.readReg32(Ctx.B, CondReg);
      Cond = Ctx.Projection.extractLaneBitFromWaveMask(Ctx.B, CondVal);
    } else {
      Cond = Ctx.Regs.loadVCC(Ctx.B);
    }
  }
  if (!Cond)
    Cond = Ctx.Regs.loadVCC(Ctx.B);
  return Cond;
}

Expected<HandlerResult> handleValuVoP3P(RaiseContext &Ctx,
                                        const DecodedInst &Di, OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;
  StringRef Mn(Di.Mnemonic);

  switch (Sop) {
  // ---- VOP3P packed ops ----
  // Handle op_sel/op_sel_hi and per-lane negation modifiers.
  case CanonicalOp::V_PK_MOV_B32: {
    Ctx.writeReg64(Op.dst(), Op.src64(0));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_FMA_F16: {
    constexpr unsigned KnownPkF16Mods = SISrcMods::NEG | SISrcMods::NEG_HI |
                                        SISrcMods::OP_SEL_0 |
                                        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (Error Err = readPackedSrcMods(Di, Op, 3, KnownPkF16Mods, Mods))
      return Err;

    int ClampIdx =
        AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx)))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", "v_pk_fma_f16 missing immediate clamp operand");

    int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
    if (ClampImm != 0 && ClampImm != 1)
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", "v_pk_fma_f16 clamp operand is not 0 or 1");

    auto *V2f16 = FixedVectorType::get(Ctx.F16Ty, 2);
    PackedSrcOptions Opts;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_f16_src";
    // VSrc_v2f16 immediates are decoded by LLVM MC as the raw 32-bit
    // packed source bits. Scalar f16 inline constants occupy the low half;
    // OP_SEL_1 controls whether the high result lane also reads that low
    // half, matching LLVM's own v_pk_fma_f16 patterns.
    Value *S0 = readPacked2Src(Ctx, Op, 0, Ctx.F16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Ctx.F16Ty, Mods[1], Opts);
    Value *S2 = readPacked2Src(Ctx, Op, 2, Ctx.F16Ty, Mods[2], Opts);
    Function *FmaFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {V2f16});
    Value *Res = Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, "pk_fma_f16");

    if (ClampImm != 0) {
      Function *MaxFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::maxnum, {V2f16});
      Function *MinFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::minnum, {V2f16});
      // AMDGPU clamp is [0, 1] after the arithmetic result; maxnum/minnum
      // gives the target-independent IR shape used elsewhere in Hotswap.
      Value *Zero = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantFP::get(Ctx.F16Ty, 0.0));
      Value *One = ConstantVector::getSplat(ElementCount::getFixed(2),
                                            ConstantFP::get(Ctx.F16Ty, 1.0));
      Res = Ctx.B.CreateCall(MaxFn, {Res, Zero}, "pk_fma_f16_clamp_lo");
      Res = Ctx.B.CreateCall(MinFn, {Res, One}, "pk_fma_f16_clamp");
    }

    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Res, Ctx.I32Ty, "pk_fma_f16_pack"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_ADD_F16:
  case CanonicalOp::V_PK_MUL_F16: {
    constexpr unsigned KnownPkF16Mods = SISrcMods::NEG | SISrcMods::NEG_HI |
                                        SISrcMods::OP_SEL_0 |
                                        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (Error Err = readPackedSrcMods(Di, Op, 2, KnownPkF16Mods, Mods))
      return Err;

    int ClampIdx =
        AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx)))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          diagnosticMnemonic(Di) + " missing immediate clamp operand");

    int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
    if (ClampImm != 0 && ClampImm != 1)
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", diagnosticMnemonic(Di) + " clamp operand is not 0 or 1");

    auto *V2f16 = FixedVectorType::get(Ctx.F16Ty, 2);
    PackedSrcOptions Opts;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_f16_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, Ctx.F16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Ctx.F16Ty, Mods[1], Opts);
    const bool IsAdd = Sop == CanonicalOp::V_PK_ADD_F16;
    const char *Name = IsAdd ? "pk_add_f16" : "pk_mul_f16";
    Value *Res =
        IsAdd ? Ctx.B.CreateFAdd(S0, S1, Name) : Ctx.B.CreateFMul(S0, S1, Name);

    if (ClampImm != 0) {
      Function *MaxFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::maxnum, {V2f16});
      Function *MinFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::minnum, {V2f16});
      Value *Zero = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantFP::get(Ctx.F16Ty, 0.0));
      Value *One = ConstantVector::getSplat(ElementCount::getFixed(2),
                                            ConstantFP::get(Ctx.F16Ty, 1.0));
      Res = Ctx.B.CreateCall(MaxFn, {Res, Zero}, Twine(Name) + "_clamp_lo");
      Res = Ctx.B.CreateCall(MinFn, {Res, One}, Twine(Name) + "_clamp");
    }

    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Res, Ctx.I32Ty, Twine(Name) + "_pack"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_ADD_BF16:
  case CanonicalOp::V_PK_MUL_BF16:
  case CanonicalOp::V_PK_MIN_NUM_BF16:
  case CanonicalOp::V_PK_MAX_NUM_BF16:
  case CanonicalOp::V_PK_FMA_BF16: {
    constexpr unsigned KnownPkBF16Mods = SISrcMods::NEG | SISrcMods::NEG_HI |
                                         SISrcMods::OP_SEL_0 |
                                         SISrcMods::OP_SEL_1;
    const bool IsFMA = Sop == CanonicalOp::V_PK_FMA_BF16;
    const bool IsMinMax = Sop == CanonicalOp::V_PK_MIN_NUM_BF16 ||
                          Sop == CanonicalOp::V_PK_MAX_NUM_BF16;
    unsigned Mods[3] = {};
    if (Error Err =
            readPackedSrcMods(Di, Op, IsFMA ? 3 : 2, KnownPkBF16Mods, Mods))
      return Err;

    int ClampIdx =
        AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx)))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          diagnosticMnemonic(Di) + " missing immediate clamp operand");

    int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
    if (ClampImm != 0 && ClampImm != 1)
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", diagnosticMnemonic(Di) + " clamp operand is not 0 or 1");

    if (ClampImm != 0 && !IsMinMax) {
      // Packed BF16 add/mul/fma do not use the ordinary VOP3 ALU clamp
      // contract ([0, 1] saturation). Their non-default clamp/overflow
      // behavior is tied to the wave's MODE.FP16_OVFL state, which this
      // raiser does not currently model. Refuse instead of silently lowering
      // it as min(max(x, 0), 1).
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          diagnosticMnemonic(Di) +
              " has a nonzero clamp bit; packed BF16 add/mul/fma clamp and "
              "overflow-mode semantics are not modelled");
    }

    Type *Bf16Ty = Type::getBFloatTy(Ctx.C);
    FixedVectorType *V2BF16 = FixedVectorType::get(Bf16Ty, 2);
    PackedSrcOptions Opts;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_bf16_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, Bf16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Bf16Ty, Mods[1], Opts);
    Value *Res = nullptr;
    if (IsFMA) {
      Value *S2 = readPacked2Src(Ctx, Op, 2, Bf16Ty, Mods[2], Opts);
      Function *FmaFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {V2BF16});
      Res = Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, "pk_fma_bf16");
    } else if (Sop == CanonicalOp::V_PK_ADD_BF16) {
      Res = Ctx.B.CreateFAdd(S0, S1, "pk_add_bf16");
    } else if (Sop == CanonicalOp::V_PK_MUL_BF16) {
      Res = Ctx.B.CreateFMul(S0, S1, "pk_mul_bf16");
    } else {
      Intrinsic::ID Id = (Sop == CanonicalOp::V_PK_MIN_NUM_BF16)
                             ? Intrinsic::minimumnum
                             : Intrinsic::maximumnum;
      Function *Fn = Intrinsic::getOrInsertDeclaration(&Ctx.M, Id, {V2BF16});
      Res = Ctx.B.CreateCall(Fn, {S0, S1},
                             Sop == CanonicalOp::V_PK_MIN_NUM_BF16
                                 ? "pk_min_num_bf16"
                                 : "pk_max_num_bf16");
    }

    if (ClampImm != 0) {
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maximumnum, {V2BF16});
      Function *MinFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minimumnum, {V2BF16});
      Value *Zero = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantFP::get(Bf16Ty, 0.0));
      Value *One = ConstantVector::getSplat(ElementCount::getFixed(2),
                                            ConstantFP::get(Bf16Ty, 1.0));
      const char *Name = (Sop == CanonicalOp::V_PK_MIN_NUM_BF16)
                             ? "pk_min_num_bf16"
                             : "pk_max_num_bf16";
      Res = Ctx.B.CreateCall(MaxFn, {Res, Zero}, Twine(Name) + "_clamp_lo");
      Res = Ctx.B.CreateCall(MinFn, {Res, One}, Twine(Name) + "_clamp");
    }

    const char *PackName = "pk_bf16_pack";
    switch (Sop) {
    case CanonicalOp::V_PK_ADD_BF16:
      PackName = "pk_add_bf16_pack";
      break;
    case CanonicalOp::V_PK_MUL_BF16:
      PackName = "pk_mul_bf16_pack";
      break;
    case CanonicalOp::V_PK_MIN_NUM_BF16:
      PackName = "pk_min_num_bf16_pack";
      break;
    case CanonicalOp::V_PK_MAX_NUM_BF16:
      PackName = "pk_max_num_bf16_pack";
      break;
    case CanonicalOp::V_PK_FMA_BF16:
      PackName = "pk_fma_bf16_pack";
      break;
    default:
      llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(Res, Ctx.I32Ty, PackName));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_PK_ADD_F32:
  case CanonicalOp::V_PK_MUL_F32:
  case CanonicalOp::V_PK_FMA_F32:
  case CanonicalOp::V_PK_MAX_F32:
  case CanonicalOp::V_PK_MIN_F32: {
    auto *V2f32 = FixedVectorType::get(Ctx.F32Ty, 2);

    constexpr unsigned KnownPkF32Mods = SISrcMods::NEG | SISrcMods::NEG_HI |
                                        SISrcMods::OP_SEL_0 |
                                        SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    unsigned NumSrcs = (Sop == CanonicalOp::V_PK_FMA_F32) ? 3 : 2;
    if (Error Err = readPackedSrcMods(Di, Op, NumSrcs, KnownPkF32Mods, Mods))
      return Err;

    // Read each source as <2 x f32>, applying source selection and negation
    // from LLVM's decoded srcN_modifiers operand.
    //
    // Two operand shapes are accepted:
    //   * Register (the common case): reads a 64-bit VGPR pair as
    //     `<2 x f32>`; lo/hi extract index the two packed lanes.
    //   * Immediate / inline literal: VOP3P encodes a single 32-bit
    //     literal per source slot which the hardware broadcasts to
    //     both packed lanes (the `op_sel_hi` modifier is ignored on
    //     scalar literals because there's only one element to choose).
    //     The swiglu tensilelite kernel exercises this path with
    //     `v_pk_add_f32 vN, vM, 0x...` where the literal is a packed
    //     bias constant.  We model it by reading the i32, bit-casting
    //     to f32, and constructing a 2-lane vector with both lanes
    //     equal to the literal.  `neg_lo` / `neg_hi` still apply per lane.
    PackedSrcOptions Opts;
    Opts.RegisterSourceIsVector = true;
    Opts.ImmediateIsScalarBroadcast = true;
    Opts.ApplyFloatNeg = true;
    Opts.Name = "pk_f32_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, Ctx.F32Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, Ctx.F32Ty, Mods[1], Opts);

    Value *Res = nullptr;
    switch (Sop) {
    case CanonicalOp::V_PK_ADD_F32:
      Res = Ctx.B.CreateFAdd(S0, S1, "pk_add");
      break;
    case CanonicalOp::V_PK_MUL_F32:
      Res = Ctx.B.CreateFMul(S0, S1, "pk_mul");
      break;
    case CanonicalOp::V_PK_MAX_F32: {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::maxnum, {V2f32});
      Res = Ctx.B.CreateCall(Fn, {S0, S1}, "pk_max");
      break;
    }
    case CanonicalOp::V_PK_MIN_F32: {
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::minnum, {V2f32});
      Res = Ctx.B.CreateCall(Fn, {S0, S1}, "pk_min");
      break;
    }
    case CanonicalOp::V_PK_FMA_F32: {
      Value *S2 = readPacked2Src(Ctx, Op, 2, Ctx.F32Ty, Mods[2], Opts);
      Function *Fn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {V2f32});
      Res = Ctx.B.CreateCall(Fn, {S0, S1, S2}, "pk_fma");
      break;
    }
    default:
      llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeRegVec(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }

  // ---- VOP3P packed-pair `<2 x i16>` int ops ----
  // Binary forms use VOP_V2I16_V2I16_V2I16; ternary forms add a third packed
  // source. Each 32-bit source is bitcast to `<2 x i16>` for the lane-wise op
  // and back to i32 for the VGPR write-back.
  case CanonicalOp::V_PK_MAD_U16:
  case CanonicalOp::V_PK_ADD_U16:
  case CanonicalOp::V_PK_SUB_I16:
  case CanonicalOp::V_PK_LSHLREV_B16:
  case CanonicalOp::V_PK_LSHRREV_B16:
  case CanonicalOp::V_PK_ASHRREV_I16:
  case CanonicalOp::V_PK_MUL_LO_U16:
  case CanonicalOp::V_PK_MAX_I16:
  case CanonicalOp::V_PK_MAX3_I16: {
    auto *I16Ty = Type::getInt16Ty(Ctx.C);
    auto *V2I16 = FixedVectorType::get(I16Ty, 2);

    constexpr unsigned KnownPkI16Mods =
        SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    const unsigned NumSrcs =
        (Sop == CanonicalOp::V_PK_MAD_U16 || Sop == CanonicalOp::V_PK_MAX3_I16)
            ? 3
            : 2;
    if (Error Err = readPackedSrcMods(Di, Op, NumSrcs, KnownPkI16Mods, Mods))
      return Err;

    PackedSrcOptions Opts;
    Opts.Name = "pk_i16_src";
    Value *S0 = readPacked2Src(Ctx, Op, 0, I16Ty, Mods[0], Opts);
    Value *S1 = readPacked2Src(Ctx, Op, 1, I16Ty, Mods[1], Opts);

    Value *Res = nullptr;
    switch (Sop) {
    case CanonicalOp::V_PK_MAD_U16: {
      Value *S2 = readPacked2Src(Ctx, Op, 2, I16Ty, Mods[2], Opts);
      int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                AMDGPU::OpName::clamp);
      if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx)))
        return RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P", "v_pk_mad_u16 missing immediate clamp operand");

      int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
      if (ClampImm != 0 && ClampImm != 1)
        return RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P", "v_pk_mad_u16 clamp operand is not 0 or 1");

      auto *V2I32 = FixedVectorType::get(Ctx.I32Ty, 2);
      Constant *Max = ConstantVector::getSplat(
          ElementCount::getFixed(2), ConstantInt::get(Ctx.I32Ty, 0xFFFFu));
      Res =
          emitU16Mad(Ctx, S0, S1, S2, ClampImm != 0, V2I32, Max, "pk_mad_u16");
      break;
    }
    case CanonicalOp::V_PK_ADD_U16:
      Res = Ctx.B.CreateAdd(S0, S1, "pk_add_u16");
      break;
    case CanonicalOp::V_PK_SUB_I16:
      // Modular i16 subtract: signed vs unsigned doesn't change the low 16
      // bits, so plain `sub` without nuw/nsw matches the AMDGPU semantics.
      Res = Ctx.B.CreateSub(S0, S1, "pk_sub_i16");
      break;
    case CanonicalOp::V_PK_MUL_LO_U16:
      // "lo" = low 16 bits of the per-lane 32-bit multiply, i.e. modular
      // u16 multiply. Plain `mul` on i16 without nuw/nsw matches that --
      // signed vs unsigned doesn't change the low half of the product.
      Res = Ctx.B.CreateMul(S0, S1, "pk_mul_lo_u16");
      break;
    case CanonicalOp::V_PK_MAX_I16: {
      Function *SmaxFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::smax, {V2I16});
      Res = Ctx.B.CreateCall(SmaxFn, {S0, S1}, "pk_max_i16");
      break;
    }
    case CanonicalOp::V_PK_MAX3_I16: {
      Value *S2 = readPacked2Src(Ctx, Op, 2, I16Ty, Mods[2], Opts);
      int ClampIdx = AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(),
                                                AMDGPU::OpName::clamp);
      if (ClampIdx < 0 || !Di.isImm(static_cast<unsigned>(ClampIdx)))
        return RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P", "v_pk_max3_i16 missing immediate clamp operand");

      int64_t ClampImm = Di.getImm(static_cast<unsigned>(ClampIdx));
      if (ClampImm != 0 && ClampImm != 1)
        return RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P", "v_pk_max3_i16 clamp operand is not 0 or 1");

      Function *SmaxFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::smax, {V2I16});
      Value *M01 = Ctx.B.CreateCall(SmaxFn, {S0, S1}, "pk_max3_i16_m01");
      Res = Ctx.B.CreateCall(SmaxFn, {M01, S2}, "pk_max3_i16");
      if (ClampImm != 0)
        Res = Ctx.B.CreateCall(SmaxFn, {Res, Constant::getNullValue(V2I16)},
                               "pk_max3_i16_clamp");
      break;
    }
    case CanonicalOp::V_PK_LSHLREV_B16: {
      // clshl_rev_16 SDAG: dst = src1 << (src0 & 15). Reversed-operand
      // convention (shift count is src0, value is src1) AND a hardware
      // clamp to the low 4 bits of the count. LLVM `shl` is poison for
      // shifts >= bitwidth, the hardware masks instead -- emit the AND
      // explicitly so the LLVM semantics match the AMDGPU semantics for
      // every legal hardware input. For constant shift counts the
      // optimiser folds the AND away; for VGPR-sourced shift counts the
      // mask is mandatory to preserve the corpus shift semantics.
      Value *Mask = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantInt::get(I16Ty, 15));
      Value *Amt = Ctx.B.CreateAnd(S0, Mask, "pk_lshlrev_amt");
      Res = Ctx.B.CreateShl(S1, Amt, "pk_lshlrev");
      break;
    }
    case CanonicalOp::V_PK_LSHRREV_B16: {
      // clshr_rev_16 SDAG: dst = src1 >>_logical (src0 & 15). Same
      // reversed-operand convention and low-4-bit hardware shift-count
      // clamp as V_PK_LSHLREV_B16 above.
      Value *Mask = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantInt::get(I16Ty, 15));
      Value *Amt = Ctx.B.CreateAnd(S0, Mask, "pk_lshrrev_amt");
      Res = Ctx.B.CreateLShr(S1, Amt, "pk_lshrrev");
      break;
    }
    case CanonicalOp::V_PK_ASHRREV_I16: {
      // cashr_rev_16 SDAG: dst = src1 >>_arith (src0 & 15). Same
      // reversed-operand convention and low-4-bit hardware shift-count
      // clamp as V_PK_LSHLREV_B16 above.
      Value *Mask = ConstantVector::getSplat(ElementCount::getFixed(2),
                                             ConstantInt::get(I16Ty, 15));
      Value *Amt = Ctx.B.CreateAnd(S0, Mask, "pk_ashrrev_amt");
      Res = Ctx.B.CreateAShr(S1, Amt, "pk_ashrrev");
      break;
    }
    default:
      llvm_unreachable("filtered by outer switch");
    }

    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(Res, Ctx.I32Ty, "pk_i16_pack"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_dot4_i32_iu8 ----
  //
  // Mixed signed/unsigned 4-byte dot product:
  //   dst = src2 + sum_{i=0..3} extA(src0.byte[i]) * extB(src1.byte[i])
  //
  // AMDGPU models the input signedness through the VOP3P source modifier
  // operands: SISrcMods::NEG set on src0/src1 means that source's packed bytes
  // are signed, otherwise they are unsigned. LLVM's `VOP3PModsNeg` pattern in
  // SIInstrInfo.td encodes the same contract. Lower to ordinary integer IR
  // rather than a target dot intrinsic so gfx1250 same-target and gfx942
  // cross-target paths share one verifier-clean semantic representation; the
  // backend may rediscover a dot instruction where legal. A future
  // target-native optimisation can route supporting targets through
  // `llvm.amdgcn.sudot4`, but that should not be required for correctness.
  case CanonicalOp::V_DOT4_I32_IU8: {
    int ClampIdx =
        AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    bool Clamp = false;
    if (ClampIdx >= 0 && Di.isImm(static_cast<unsigned>(ClampIdx)))
      Clamp = Di.getImm(static_cast<unsigned>(ClampIdx)) != 0;
    if (ClampIdx >= 0 && !Di.isImm(static_cast<unsigned>(ClampIdx)))
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", "v_dot4_i32_iu8 clamp operand is not an immediate");

    Value *Src0 = Op.src(0);
    Value *Src1 = Op.src(1);
    Value *Acc = Ctx.B.CreateSExt(Op.src(2), Ctx.I64Ty, "dot4_acc_wide");
    auto *I8Ty = Type::getInt8Ty(Ctx.C);

    auto ExtendByte = [&](Value *Packed, unsigned ByteIdx,
                          bool IsSigned) -> Value * {
      Value *Shift = ConstantInt::get(Ctx.I32Ty, ByteIdx * 8);
      Value *Lo =
          Ctx.B.CreateTrunc(Ctx.B.CreateLShr(Packed, Shift), I8Ty, "dot4_byte");
      return IsSigned ? Ctx.B.CreateSExt(Lo, Ctx.I64Ty, "dot4_sext")
                      : Ctx.B.CreateZExt(Lo, Ctx.I64Ty, "dot4_zext");
    };

    const bool Src0Signed = (Op.srcMod(0) & SISrcMods::NEG) != 0;
    const bool Src1Signed = (Op.srcMod(1) & SISrcMods::NEG) != 0;
    for (unsigned I = 0; I < 4; ++I) {
      Value *A = ExtendByte(Src0, I, Src0Signed);
      Value *B = ExtendByte(Src1, I, Src1Signed);
      Acc = Ctx.B.CreateAdd(Acc, Ctx.B.CreateMul(A, B, "dot4_mul"), "dot4_acc");
    }

    if (Clamp) {
      Value *Lo = ConstantInt::get(Ctx.I64Ty, INT32_MIN);
      Value *Hi = ConstantInt::get(Ctx.I64Ty, INT32_MAX);
      Acc = Ctx.B.CreateSelect(Ctx.B.CreateICmpSLT(Acc, Lo), Lo, Acc,
                               "dot4_clamp_lo");
      Acc = Ctx.B.CreateSelect(Ctx.B.CreateICmpSGT(Acc, Hi), Hi, Acc,
                               "dot4_clamp");
    }

    Ctx.writeReg32(Op.dst(), Ctx.B.CreateTrunc(Acc, Ctx.I32Ty, "dot4_i32"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- WMMA (gfx1250 RDNA4, VOP3P encoding) ----
  // 16x16xK WMMA family. Three K-families x accumulator-type
  // permutations covered today:
  //   * 16-bit elements, K=32, f32 acc (8 VGPRs of <16 x t> per A/B side):
  //       v_wmma_f32_16x16x32_f16,  v_wmma_f32_16x16x32_bf16
  //   * 8-bit elements,  K=64, f32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_f32_16x16x64_<a>_<b>  for a,b in {fp8, bf8}
  //   * 8-bit elements,  K=64, i32 acc (8 VGPRs of <8 x i32> per A/B side):
  //       v_wmma_i32_16x16x64_iu8  (signed/unsigned 8-bit integer GEMMs)
  //
  // All share the per-Wave32-lane A/B fragment shape (8 VGPRs, 32 bytes).
  // The C/D side is <8 x f32> for the f32-accumulator variants and
  // <8 x i32> for the IU8 integer-accumulator variant. The WMMA12
  // native-intrinsic path (when target supports it) and the gfx942
  // MFMA lowering path (`emitWMMAtoMFMA`, parameterised on
  // `WMMAInputType`) are uniform across the entire family -- the local
  // A/B IR vector type + native-WMMA intrinsic ID + WMMAInputType +
  // accumulator IR type is the only delta between variants. "Design
  // the operation, not the opcode."
  //
  // Native WMMA12 intrinsic-call shapes split THREE ways:
  //   * 16-bit f32-acc: AMDGPUWmmaIntrinsicModsAllReuse -- 8 args
  //       (A_mod, A, B_mod, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  f32-acc: AMDGPUWmmaIntrinsicModsC       -- 6 args
  //       (A, B, C_mod, C, reuse_a, reuse_b)
  //   * 8-bit  i32-acc: AMDGPUWmmaIntrinsicModsABClamp -- 8 args
  //       (A_mod, A, B_mod, B, C, reuse_a, reuse_b, clamp)
  // The MFMA fallback path is uniform across all three.
  // 16x16x4 WMMA (32-bit f32 A/B/C, gfx1250 VOP3P opcode 0x05D).
  // This handler stands alone from the K=32 / K=64 family below
  // because (a) the per-lane A/B fragment is `<2 x f32>` (only 2
  // dwords) instead of <16 x t> (16-bit) or <8 x i32> (8-bit), and
  // (b) `emitWMMAtoMFMA` is parameterised on 16-/8-bit element
  // packing and does not cover the K=4 f32 case.
  //
  // The native intrinsic `int_amdgcn_wmma_f32_16x16x4_f32` is
  // declared inside `AMDGPUWMMAIntrinsicsGFX1250` (gated by
  // `isGFX125xOnly` in IntrinsicsAMDGPU.td:4113-4114), so it is
  // strictly gfx1250-only -- the gfx12 (RDNA4 base) WMMA family
  // (`AMDGPUWMMAIntrinsicsGFX12`, gated by `hasWMMA12` =
  // FeatureWMMA{128,256}bInsts) does NOT include it. Same-target
  // lift therefore gates on `Ctx.TargetIsa.hasTensorOps`
  // (FeatureGFX1250Insts), not `hasWMMA12`.
  //
  // Cross-target on gfx942 we lower to `mfma_f32_16x16x4f32` via the
  // dedicated `emitWMMAtoMFMA_F32_16x16x4` helper in
  // `wmma-lowering.cpp` -- gfx942 has a direct K=4 MFMA equivalent
  // so the decomposition is 1 MFMA per Wave32 group (not 2 chained
  // like the K=32/K=64 path). The shared ds_bpermute redistribution
  // math is documented alongside the helper. Targets with neither
  // `hasTensorOps` nor `hasMFMA` (e.g. gfx12 RDNA4 base) get a
  // principled refusal -- they have no K=4 f32 matrix path at all.
  case CanonicalOp::V_WMMA_F32_16x16x4_F32: {
    auto *AbIrTy = FixedVectorType::get(Ctx.F32Ty, 2);
    auto *CdIrTy = FixedVectorType::get(Ctx.F32Ty, 8);

    ParsedReg Dest = Op.dst();
    ParsedReg SrcA = Op.srcReg(0), SrcB = Op.srcReg(1);

    Value *A = Ctx.Regs.readRegVec(Ctx.B, SrcA, AbIrTy);
    Value *B = Ctx.Regs.readRegVec(Ctx.B, SrcB, AbIrTy);
    Expected<Value *> C = readWMMAAccumC(Ctx, Di, Op, Dest, CdIrTy);
    if (!C)
      return C.takeError();

    Value *ResultVal;
    if (Ctx.TargetIsa.HasTensorOps) {
      Function *WmmaFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_wmma_f32_16x16x4_f32, {CdIrTy, AbIrTy});
      // AMDGPUWmmaIntrinsicModsC (6 args, see IntrinsicsAMDGPU.td):
      //   (A, B, C_mod, C, matrix_a_reuse, matrix_b_reuse)
      // C_mod is the i16 source-modifier bitfield (op_sel etc.) and
      // matrix_*_reuse are i1 flags. K=4 f32 WMMA has NO per-element
      // A_mod / B_mod slots (unlike the 16-/8-bit ModsAllReuse /
      // ModsABClamp classes used by the K=32 / K=64 family). The
      // gfx1250 corpus emits the instruction without those modifiers
      // set; defaulting to 0 / false matches what the disassembler
      // surfaces for the failing kernels.
      ResultVal =
          Ctx.B.CreateCall(WmmaFn,
                           {A, B, ConstantInt::get(Type::getInt16Ty(Ctx.C), 0),
                            *C, Ctx.B.getFalse(), Ctx.B.getFalse()},
                           "wmma");
    } else if (Ctx.TargetIsa.HasMfma) {
      // Same-shape gate as the K=32/K=64 case below.  The staged
      // strict.wwm-scoped MODREP lowering is verified correct for
      // minimal-repro kernels (isolated and K-loop-chained WMMAs)
      // but an unexplained residual divergence remains on the
      // Triton `matmul_fp16_16x16` kernel at M>=32 through
      // `compare_correctness`.  See the K=32/K=64 arm below for
      // the full discussion.  Gate stays in place until the
      // residual is pinned and the fix lands; the infrastructure
      // in `wave-projection.h` (`numSourceWavesPerTarget`,
      // `wrapAsWWMValue`) is LANDED additively.
      // K=4 f32 arm: previously conservatively refused under MODREP
      // when a multi-WMMA-per-K-iter pattern (permlane16_swap
      // presence) was detected.  The root cause turned out to be a
      // wrong-semantic lift of `v_permlane16_swap_b32` (symmetric
      // vs. ISA-asymmetric -- see `handle-valu-cross-lane.cpp` and
      // matrix-translation.md sec. 12.4.7), not a WMMA-lowering
      // problem, so with that fixed the MODREP MFMA lowering
      // handles both single- and multi-WMMA cases correctly.
      {
        Expected<Value *> RV = emitWmmAtoMfmaF3216x16x4(Ctx, A, B, *C);
        if (!RV)
          return RV.takeError();
        ResultVal = *RV;
      }
    } else {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P",
          "v_wmma_f32_16x16x4_f32 cross-target requires either "
          "hasTensorOps (native gfx1250 intrinsic "
          "int_amdgcn_wmma_f32_16x16x4_f32) or hasMFMA (gfx942 "
          "mfma_f32_16x16x4f32 decomposition); this target has "
          "neither -- no K=4 f32 matrix path is available");
    }

    Ctx.writeRegVec(Dest, ResultVal);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_fma_mixlo_bf16: BF16-result mixed-precision FMA (VOP3P) ----
  //
  // LLVM's TableGen definitions declare V_FMA_MIX{LO,HI}_{F16,BF16} with
  // FPDPRounding=1. The generated selection patterns model them as:
  //
  //   fptrunc_narrow(llvm.fma.f32(cvt_f32(src0_part),
  //                               cvt_f32(src1_part),
  //                               cvt_f32(src2_part)))
  //
  // and the ISA family writes only one 16-bit half of vdst (the other half is
  // the tied vdst_in input). The source `*_part` selection matches
  // V_FMA_MIX_F32{,_BF16} below: op_sel_hi chooses narrow vs full-f32, and
  // op_sel chooses the high half when a register source is interpreted as
  // narrow. LO/HI differ only in destination half; F16/BF16 differ only in
  // narrow type.
  case CanonicalOp::V_FMA_MIXLO_F16:
  case CanonicalOp::V_FMA_MIXHI_F16:
  case CanonicalOp::V_FMA_MIXLO_BF16:
  case CanonicalOp::V_FMA_MIXHI_BF16: {
    StringRef InstrName = diagnosticMnemonic(Di);
    if (Op.nSrcs() < 3)
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP3P", InstrName + " requires three explicit source operands");

    bool IsBF16 = Sop == CanonicalOp::V_FMA_MIXLO_BF16 ||
                  Sop == CanonicalOp::V_FMA_MIXHI_BF16;
    bool WritesHigh = Sop == CanonicalOp::V_FMA_MIXHI_F16 ||
                      Sop == CanonicalOp::V_FMA_MIXHI_BF16;
    Type *NarrowTy = IsBF16 ? Type::getBFloatTy(Ctx.C) : Ctx.F16Ty;
    const char *CvtName =
        IsBF16 ? (WritesHigh ? "mixhi_cvt_bf16" : "mixlo_cvt_bf16")
               : (WritesHigh ? "mixhi_cvt" : "mixlo_cvt");
    const char *FMAName =
        IsBF16 ? (WritesHigh ? "fma_mixhi_bf16" : "fma_mixlo_bf16")
               : (WritesHigh ? "fma_mixhi_f16" : "fma_mixlo_f16");

    bool ClampResult = false;
    int ClampIndex =
        AMDGPU::getNamedOperandIdx(Di.Inst.getOpcode(), AMDGPU::OpName::clamp);
    if (ClampIndex >= 0) {
      if (!Di.isImm(static_cast<unsigned>(ClampIndex)))
        return RaiseFailure::unsupportedInstructionForm(
            Di, "VOP3P", InstrName + " clamp operand is not an immediate");

      ClampResult = Di.getImm(static_cast<unsigned>(ClampIndex)) != 0;
    }

    Type *I16Ty = Type::getInt16Ty(Ctx.C);

    constexpr unsigned KnownMixMods = SISrcMods::NEG | SISrcMods::ABS |
                                      SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (Error Err = readSourceMods(Di, Op, 3, KnownMixMods, Mods))
      return Err;

    Value *S0 = readMixF32Src(Ctx, Op, 0, NarrowTy, Mods[0], CvtName);
    Value *S1 = readMixF32Src(Ctx, Op, 1, NarrowTy, Mods[1], CvtName);
    Value *S2 = readMixF32Src(Ctx, Op, 2, NarrowTy, Mods[2], CvtName);
    Function *FmaFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Value *Fma = Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, FMAName);
    Value *Rounded =
        Ctx.B.CreateFPTrunc(Fma, NarrowTy, (Twine(FMAName) + "_round").str());
    if (ClampResult) {
      // AMDGPUclamp clamps to [0, 1] and maps NaN to 0 (SIInstrInfo.td).
      // V_FMA_MIX{LO,HI} applies it after destination narrow-type rounding.
      Function *MaxFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::maxnum, {NarrowTy});
      Function *MinFn = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::minnum, {NarrowTy});
      Rounded = Ctx.B.CreateCall(
          MinFn,
          {Ctx.B.CreateCall(MaxFn, {Rounded, ConstantFP::get(NarrowTy, 0.0)},
                            (Twine(FMAName) + "_clamp_lo").str()),
           ConstantFP::get(NarrowTy, 1.0)},
          (Twine(FMAName) + "_clamp").str());
    }
    Value *NarrowBits =
        Ctx.B.CreateZExt(Ctx.B.CreateBitCast(Rounded, I16Ty), Ctx.I32Ty);

    ParsedReg Dest = Op.dst();
    Value *OldDest = Ctx.Regs.readReg32(Ctx.B, Dest);
    if (WritesHigh) {
      Value *OldLo =
          Ctx.B.CreateAnd(OldDest, ConstantInt::get(Ctx.I32Ty, 0x0000FFFFu),
                          (Twine(FMAName) + "_old_lo").str());
      Value *HiBits =
          Ctx.B.CreateShl(NarrowBits, 16, (Twine(FMAName) + "_hi_bits").str());
      Ctx.writeReg32(Dest, Ctx.B.CreateOr(OldLo, HiBits,
                                          (Twine(FMAName) + "_pack").str()));
    } else {
      Value *OldHi =
          Ctx.B.CreateAnd(OldDest, ConstantInt::get(Ctx.I32Ty, 0xFFFF0000u),
                          (Twine(FMAName) + "_old_hi").str());
      Ctx.writeReg32(Dest, Ctx.B.CreateOr(OldHi, NarrowBits,
                                          (Twine(FMAName) + "_pack").str()));
    }
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_fma_mix_f32 / v_fma_mix_f32_bf16: mixed-precision FMA (VOP3P) ----
  //
  // dst = fma(cvt_f32(src0_part), cvt_f32(src1_part), cvt_f32(src2_part))
  //
  // Per-source selection is driven by the VOP3P op_sel / op_sel_hi
  // modifier pair carried in LLVM's decoded srcN_modifiers operands:
  //
  //   op_sel_hi[i]==0  -> source i is the full f32 VGPR
  //   op_sel_hi[i]==1  -> source i is the 16-bit half selected by
  //                       op_sel[i] (0=lo [15:0], 1=hi [31:16])
  //                       interpreted as the mnemonic's narrow type
  //                       (f16 for V_FMA_MIX_F32, bf16 for
  //                       V_FMA_MIX_F32_BF16), then fpext'd to f32.
  //
  // The BF16 variant does NOT need a cross-target refusal because
  // `fpext bfloat to float` is universally lowered (it's a shift-left-16
  // + bitcast on every AMDGPU target); only the narrow element type
  // switches.
  case CanonicalOp::V_FMA_MIX_F32:
  case CanonicalOp::V_FMA_MIX_F32_BF16: {
    Type *NarrowTy = (Sop == CanonicalOp::V_FMA_MIX_F32_BF16)
                         ? Type::getBFloatTy(Ctx.C)
                         : Ctx.F16Ty;
    const char *CvtName =
        (Sop == CanonicalOp::V_FMA_MIX_F32_BF16) ? "mix_cvt_bf16" : "mix_cvt";

    constexpr unsigned KnownMixMods = SISrcMods::NEG | SISrcMods::ABS |
                                      SISrcMods::OP_SEL_0 | SISrcMods::OP_SEL_1;
    unsigned Mods[3] = {};
    if (Error Err = readSourceMods(Di, Op, 3, KnownMixMods, Mods))
      return Err;

    // `OP_SEL_0` is a VGPR-half selector and only makes sense when the source
    // is a 32-bit VGPR that holds two packed 16-bit values. For immediates,
    // LLVM's AMDGPU disassembler pre-resolves narrow-width operands to the
    // 16-bit value in the low half of the MCOperand immediate; the helper
    // therefore ignores OP_SEL_0 for non-register narrow sources.
    Value *S0 = readMixF32Src(Ctx, Op, 0, NarrowTy, Mods[0], CvtName);
    Value *S1 = readMixF32Src(Ctx, Op, 1, NarrowTy, Mods[1], CvtName);
    Value *S2 = readMixF32Src(Ctx, Op, 2, NarrowTy, Mods[2], CvtName);
    Function *FmaFn =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::fma, {Ctx.F32Ty});
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(FmaFn, {S0, S1, S2}, "fma_mix"),
                            Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_cndmask_b32 (VOP2 or VOP3 -- srcMap skips modifiers) ----
  case CanonicalOp::V_CNDMASK_B32: {
    ParsedReg Dest = Op.dst();
    Value *Src0 = Op.applyMods(0, Op.src(0));
    Value *Src1 = Op.applyMods(1, Op.src(1));
    if (Src0->getType() == Ctx.F32Ty)
      Src0 = Ctx.B.CreateBitCast(Src0, Ctx.I32Ty);
    if (Src1->getType() == Ctx.F32Ty)
      Src1 = Ctx.B.CreateBitCast(Src1, Ctx.I32Ty);
    Value *Cond = raiseCndmaskWaveCondition(Ctx, Di, Op);
    Ctx.writeReg32(Dest, Ctx.B.CreateSelect(Cond, Src1, Src0, "cndmask"));
    Hr.Handled = true;
    return Hr;
  }

  default:
    break;
  }
  return Hr;
}

} // namespace COMGR::hotswap
