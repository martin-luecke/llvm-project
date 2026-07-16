//===- handle-valu-small-ops.cpp - Hotswap transpiler ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-f16-utils.h"
#include "handle-valu-internal.h"
#include "handle-valu-output-mods.h"

#include "canonical-op.h"
#include "ocml-runtime.h"

#include "SIDefines.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/LogicalResult.h"

using namespace llvm;

namespace COMGR::hotswap {

// "Small ops": F32<->{U,I}32 conversions, 16-bit reverse-operand shifts,
// register-relative moves, bit ops (V_NOT_B32 / V_BFREV_B32 / V_FFBH_U32 /
// V_FFBH_I32 / V_FFBL_B32 / V_PACK_B32_F16 / V_PRNG_B32), and the F32
// single-src transcendentals the corpus needs (rcp/exp/rsq/rndne, plus the
// scalar s_rcp/s_rsq and rcp_iflag).
//
// Grouped here because each case is 1-5 lines of IR emission and they
// would bloat the arithmetic / 3-src sub-handlers if interleaved.
// Structured as a switch on CanonicalOp: cases are mutually exclusive and
// ordering is not load-bearing.
Expected<HandlerResult>
handleValuSmallOps(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  HandlerResult Hr;
  Type *I16Ty = Type::getInt16Ty(Ctx.C);

  switch (Di.CanonOp) {
  // ---- Register-relative moves (v_movrel{d,s,sd}_b32) ----
  //
  // These access a VGPR at an M0-relative index. M0 is uniform across
  // lanes, so there is no cross-lane component -- this is a plain indexed
  // register access. Because the reg file promotes VGPRs to SSA by index,
  // the M0-relative index must be resolved at raise time; we use the M0
  // constant shadow (RaiseContext::getM0Const), which covers the common
  // unrolled-copy-loop idiom (e.g. CatArrayBatchedCopy). A data-dependent
  // M0 has no statically-known index and is refused loudly (stubbed).
  //
  //   v_movreld_b32  vdst, vsrc : VGPR[base(vdst)+M0] = vsrc  (tied vdst_in)
  //   v_movrels_b32  vdst, vsrc : vdst = VGPR[base(vsrc)+M0]
  //   v_movrelsd_b32 vdst, vsrc : VGPR[base(vdst)+M0] = VGPR[base(vsrc)+M0]
  case CanonicalOp::V_MOVRELD_B32:
  case CanonicalOp::V_MOVRELS_B32:
  case CanonicalOp::V_MOVRELSD_B32: {
    unsigned Opc = Di.Inst.getOpcode();
    int VdstIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::vdst);
    int VsrcIdx = AMDGPU::getNamedOperandIdx(Opc, AMDGPU::OpName::src0);
    if (VdstIdx < 0 || VsrcIdx < 0 || !Di.isReg(VdstIdx) ||
        !Di.isReg(VsrcIdx)) {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1", "v_movrel* missing vdst/vsrc register operand");
    }
    std::optional<uint64_t> M0 = Ctx.getM0Const();
    if (!M0) {
      // Data-dependent M0: no statically-known relative index. Refuse
      // rather than emit an unbounded index cascade.
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "v_movrel* with non-constant M0 (data-dependent register-relative "
          "index) is not supported; only a raise-time-constant M0 is handled");
    }
    ParsedReg VdstBase = Ctx.parseReg(Di.getReg(VdstIdx), VdstIdx);
    ParsedReg VsrcBase = Ctx.parseReg(Di.getReg(VsrcIdx), VsrcIdx);
    auto InRange = [](long Idx) {
      return Idx >= 0 && Idx < static_cast<long>(AllocaRegFile::KVGPRCap);
    };
    assert(*M0 <= UINT32_MAX && "M0 is a 32-bit hardware register");
    long Rel = static_cast<long>(*M0);
    bool RelDst = Di.CanonOp == CanonicalOp::V_MOVRELD_B32 ||
                  Di.CanonOp == CanonicalOp::V_MOVRELSD_B32;
    bool RelSrc = Di.CanonOp == CanonicalOp::V_MOVRELS_B32 ||
                  Di.CanonOp == CanonicalOp::V_MOVRELSD_B32;
    long DstIdx = VdstBase.BaseIdx + (RelDst ? Rel : 0);
    long SrcIdx = VsrcBase.BaseIdx + (RelSrc ? Rel : 0);
    if (!InRange(DstIdx) || !InRange(SrcIdx)) {
      return RaiseFailure::unsupportedInstructionForm(
          Di, "VOP1",
          "v_movrel* M0-relative VGPR index out of range (MEMVIOL)");
    }
    // Read the value to move: vsrc's SSA value for V_MOVRELD; the
    // relative-source VGPR for V_MOVRELS / V_MOVRELSD.
    Value *Val = RelSrc ? Ctx.Regs.loadVGPR32(Ctx.B, static_cast<int>(SrcIdx))
                        : Ctx.readOp32(Di, static_cast<unsigned>(VsrcIdx));
    ParsedReg DstPr;
    DstPr.RegKind = ParsedReg::VGPR;
    DstPr.BaseIdx = static_cast<int>(DstIdx);
    DstPr.WidthInDwords = 1;
    Ctx.writeReg32(DstPr, Val);
    Hr.Handled = true;
    return Hr;
  }
  // ---- F32 <-> integer conversions ----
  case CanonicalOp::V_CVT_F32_U32: {
    Value *R = Ctx.B.CreateUIToFP(Op.src(0), Ctx.F32Ty, "cvt");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_F32_I32: {
    Value *R = Ctx.B.CreateSIToFP(Op.src(0), Ctx.F32Ty, "cvt");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_U32_F32: {
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateFPToUI(S, Ctx.I32Ty, "cvt"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_CVT_I32_F32: {
    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateFPToSI(S, Ctx.I32Ty, "cvt"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- 16-bit reverse-operand shifts (HW uses src0[3:0]) ----
  case CanonicalOp::V_ASHRREV_I16:
  case CanonicalOp::V_LSHRREV_B16:
  case CanonicalOp::V_LSHLREV_B16: {
    Value *Shamt = Ctx.B.CreateAnd(Ctx.B.CreateTrunc(Op.src(0), I16Ty),
                                   ConstantInt::get(I16Ty, 0xF));
    Value *Base = Ctx.B.CreateTrunc(Op.src(1), I16Ty);
    Value *Res = nullptr;
    switch (Di.CanonOp) {
    case CanonicalOp::V_ASHRREV_I16:
      Res = Ctx.B.CreateAShr(Base, Shamt, "vashr16");
      break;
    case CanonicalOp::V_LSHRREV_B16:
      Res = Ctx.B.CreateLShr(Base, Shamt, "vlshr16");
      break;
    case CanonicalOp::V_LSHLREV_B16:
      Res = Ctx.B.CreateShl(Base, Shamt, "vlshl16");
      break;
    default:
      llvm_unreachable("filtered by outer switch");
    }
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateZExt(Res, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }

  case CanonicalOp::V_PACK_B32_F16: {
    Value *Lo = Ctx.B.CreateAnd(Op.src(0), ConstantInt::get(Ctx.I32Ty, 0xFFFF));
    Value *Hi = Ctx.B.CreateShl(
        Ctx.B.CreateAnd(Op.src(1), ConstantInt::get(Ctx.I32Ty, 0xFFFF)), 16);
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateOr(Lo, Hi, "pack_f16"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- Simple bit-twiddle single-src ----
  case CanonicalOp::V_BFREV_B32: {
    Function *Brev = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::bitreverse, {Ctx.I32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(Brev, {Op.src(0)}, "bfrev"));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_NOT_B32: {
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateNot(Op.src(0), "vnot"));
    Hr.Handled = true;
    return Hr;
  }

  // ---- find-first-bit (VOP1, gfx7+) ----
  // V_FFBH_U32 / V_FFBL_B32 use llvm.ctlz / llvm.cttz with
  // is_zero_undef=false so LLVM returns the bitwidth (32) for input 0.
  // Hardware instead returns -1 for input 0, so we explicitly cmov to
  // -1 on the zero-input path. V_FFBH_I32 uses the dedicated
  // llvm.amdgcn.sffbh intrinsic which selects directly back to
  // v_ffbh_i32_e32 (no fixup needed -- the intrinsic and the hardware
  // share the "-1 on uniform-sign input" convention).
  case CanonicalOp::V_FFBH_U32: {
    Function *Ctlz =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::ctlz, {Ctx.I32Ty});
    Value *Src = Op.src(0);
    Value *Raw = Ctx.B.CreateCall(Ctlz, {Src, ConstantInt::getFalse(Ctx.I1Ty)},
                                  "ffbh_u32_raw");
    Value *IsZero = Ctx.B.CreateICmpEQ(Src, Ctx.B.getInt32(0), "ffbh_u32_zero");
    Value *Res =
        Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Raw, "ffbh_u32");
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FFBL_B32: {
    Function *Cttz =
        Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::cttz, {Ctx.I32Ty});
    Value *Src = Op.src(0);
    Value *Raw = Ctx.B.CreateCall(Cttz, {Src, ConstantInt::getFalse(Ctx.I1Ty)},
                                  "ffbl_b32_raw");
    Value *IsZero = Ctx.B.CreateICmpEQ(Src, Ctx.B.getInt32(0), "ffbl_b32_zero");
    Value *Res =
        Ctx.B.CreateSelect(IsZero, Ctx.B.getInt32(-1), Raw, "ffbl_b32");
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_FFBH_I32: {
    Function *Sffbh = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_sffbh, {Ctx.I32Ty});
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateCall(Sffbh, {Op.src(0)}, "ffbh_i32"));
    Hr.Handled = true;
    return Hr;
  }

  // out = (in << 1) ^ (in[31] ? 197 : 0). Use the intrinsic where it
  // selects (HasPrngInst); expand in IR for targets without a pattern.
  case CanonicalOp::V_PRNG_B32: {
    Value *Src = Op.src(0);
    Value *Res;
    if (Ctx.TargetIsa.HasPrngInst) {
      Function *PrngFn =
          Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_prng_b32);
      Res = Ctx.B.CreateCall(PrngFn, {Src}, "prng_b32");
    } else {
      Value *Shl =
          Ctx.B.CreateShl(Src, ConstantInt::get(Ctx.I32Ty, 1), "prng_shl");
      Value *Neg =
          Ctx.B.CreateICmpSLT(Src, ConstantInt::get(Ctx.I32Ty, 0), "prng_neg");
      Value *Tap =
          Ctx.B.CreateSelect(Neg, ConstantInt::get(Ctx.I32Ty, 197),
                             ConstantInt::get(Ctx.I32Ty, 0), "prng_tap");
      Res = Ctx.B.CreateXor(Shl, Tap, "prng_b32");
    }
    Ctx.writeReg32(Op.dst(), Res);
    Hr.Handled = true;
    return Hr;
  }

  // ---- F32 scalar math / rounding ----
  case CanonicalOp::V_RCP_IFLAG_F32: {
    if (Error Err = requireDefaultOutputModsIfPresent(Di))
      return Err;

    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Value *R = Ctx.B.CreateFDiv(ConstantFP::get(Ctx.F32Ty, 1.0), S, "rcp");
    Ctx.writeReg32(Op.dst(), Ctx.B.CreateBitCast(R, Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RCP_F32:
  case CanonicalOp::V_S_RCP_F32: {
    if (Di.CanonOp == CanonicalOp::V_S_RCP_F32)
      if (Error Err = requireDefaultPseudoScalarOutputMods(Di))
        return Err;

    if (Di.CanonOp == CanonicalOp::V_RCP_F32)
      if (Error Err = requireDefaultOutputModsIfPresent(Di))
        return Err;

    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *RcpFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_rcp, {Ctx.F32Ty});
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(RcpFn, {S}, "rcp"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_EXP_F32: {
    if (Error Err = requireDefaultOutputModsIfPresent(Di))
      return Err;

    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *Exp2Fn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_exp2, {Ctx.F32Ty});
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(Exp2Fn, {S}, "exp"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RSQ_F32:
  case CanonicalOp::V_S_RSQ_F32: {
    if (Di.CanonOp == CanonicalOp::V_S_RSQ_F32)
      if (Error Err = requireDefaultPseudoScalarOutputMods(Di))
        return Err;

    if (Di.CanonOp == CanonicalOp::V_RSQ_F32)
      if (Error Err = requireDefaultOutputModsIfPresent(Di))
        return Err;

    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *RsqFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_rsq, {Ctx.F32Ty});
    Ctx.writeReg32(
        Op.dst(),
        Ctx.B.CreateBitCast(Ctx.B.CreateCall(RsqFn, {S}, "rsq"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_RNDNE_F32: {
    if (Error Err = requireDefaultOutputModsIfPresent(Di))
      return Err;

    Value *S = Ctx.B.CreateBitCast(Op.srcF(0), Ctx.F32Ty);
    Function *RoundEvenFn = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::roundeven, {Ctx.F32Ty});
    Ctx.writeReg32(Op.dst(),
                   Ctx.B.CreateBitCast(
                       Ctx.B.CreateCall(RoundEvenFn, {S}, "rndne"), Ctx.I32Ty));
    Hr.Handled = true;
    return Hr;
  }
  default:
    break;
  }
  return Hr;
}

} // namespace COMGR::hotswap
