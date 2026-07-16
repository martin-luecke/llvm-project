//===- handle-valu-cross-lane.cpp - Hotswap transpiler --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "handle-valu-internal.h"

#include "canonical-op.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::OpName
#include "SIDefines.h"                       // SISrcMods::OP_SEL_0
#include "Utils/AMDGPUBaseInfo.h"

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace COMGR::hotswap {

// Cross-lane VALU primitives -- the subset of VALU opcodes whose result
// in lane L depends on values held by lane L' != L. Isolated from the
// rest of handleVALU because this is exactly the surface the cross-
// wave strategy (hotswap/docs/wave-size-translation.md sec. sec. 5.3 and 7)
// keeps iterating on: every rewrite from the "wave-size-baked cross-
// lane" rewrite table lands in this file, not scattered through the
// VALU arithmetic sections.
//
// Each branch MUST use a genuine cross-lane LLVM intrinsic
// (`llvm.amdgcn.readlane`, `writelane`, `readfirstlane`, `mbcnt.{lo,
// hi}`, etc.). A "same-lane" stub that ignores the source-lane
// selector is a silent miscompile for any kernel that feeds divergent
// operands into the primitive. Several permlane variants here are
// known broken (see the pending-rewrite table in wave-size-
// translation.md sec. 7); they stay same-lane for now but any new cross-
// lane CanonicalOp must be modelled correctly before landing.

Expected<HandlerResult>
handleValuCrossLane(RaiseContext &Ctx, const DecodedInst &Di, OpResolver &Op) {
  HandlerResult Hr;
  CanonicalOp Sop = Di.CanonOp;

  switch (Sop) {

  // ---- v_permlane16_b32 / v_permlanex16_b32 ----
  // P2 lowering -- see the permlane16 / permlanex16 row of hotswap/
  // docs/wave-size-translation.md sec. 5.3. Target constraint: `v_permlane16`
  // and `v_permlanex16` are RDNA/gfx10+ instructions and DO NOT exist
  // on CDNA (gfx9/gfx94x). Emitting `llvm.amdgcn.permlane16` or
  // `permlanex16` directly fails isel on gfx942 with "Cannot select:
  // intrinsic %llvm.amdgcn.permlanex16". We therefore emulate both
  // via `ds_bpermute_b32`, which IS available on every AMDGPU
  // generation with LDS (gfx8+), so this lowering is target-
  // independent -- it works for gfx1250 -> gfx942, gfx1250 -> gfx1250,
  // and any future target with ds_bpermute.
  //
  // MCInst operand layout (from VOP3_PERMLANE_Profile's InsVOP3OpSel):
  //
  //   [0] vdst (output)             [5] src2_modifiers (always 0)
  //   [1] src0_modifiers  <-- fi    [6] src2 (SSrc_b32)  = selector_2
  //   [2] src0 (VRegSrc_32) = val   [7] vdst_in (VGPR, tied) = %old
  //   [3] src1_modifiers  <-- bc    [8] op_sel (VOP3OpSel imm, unused
  //   [4] src1 (SSrc_b32)  = sel_1                                here)
  //
  // Selector encoding: src1 and src2 are each 32-bit scalar values
  // containing 8 x 4-bit per-lane selectors. src1 covers within-
  // group lanes 0..7, src2 covers within-group lanes 8..15. Each
  // 4-bit nibble selects a source lane within the 16-lane group.
  //
  // Per-lane emulation, L = `mbcnt`-derived absolute lane id (0..W_t):
  //
  //   group_base  = L & ~0xF           // 16, 32, 48 boundaries
  //   within      = L & 0xF            // 0..15
  //   within_lo   = within & 7         // 0..7 (nibble index)
  //   sel_word    = within < 8 ? src1 : src2
  //   nibble      = (sel_word >> (within_lo * 4)) & 0xF
  //
  //   permlane16  : src_group = group_base
  //   permlanex16 : src_group = group_base ^ 0x10  (swap adjacent groups)
  //
  //   src_lane_abs = src_group | nibble
  //   byte_addr    = src_lane_abs << 2
  //   result       = ds_bpermute(byte_addr, src0)
  //
  // Wave-width correctness under modulo-replication (hotswap/docs/
  // wave-size-translation.md sec. 6's wave-size-obliviousness theorem):
  // the source gfx1250 kernel is wave32 so its selector values
  // encode a shuffle pattern over 2 x 16-lane groups. On wave64
  // target each modrep replica occupies 2 x 16-lane groups (R=2),
  // and the `group ^ 0x10` swap stays within a replica (0<->1 within
  // replica 0, 2<->3 within replica 1), so the modrep invariant is
  // preserved for permlanex16. permlane16 keeps every lane within
  // its own group, trivially within-replica.
  //
  // Handling of `fi` (fetch-invalid) and `bc` (bound_ctrl) -- the two
  // i1 immediates encoded via `opsel_i1timm` in PermlanePat
  // (`SISrcMods::OP_SEL_0` bit of src0_modifiers / src1_modifiers):
  //
  //   - `fi=1`: on an EXEC-inactive source lane, the kernel still
  //     fetches that lane's VGPR value (possibly stale). This is
  //     exactly how `llvm.amdgcn.ds.bpermute` behaves naturally
  //     (the LDS-backed path reads the VGPR alloca regardless of
  //     EXEC), so `fi=1` is supported directly.
  //   - `bc=0`: on an "out-of-range" source lane, the target lane
  //     retains %old. For permlane16 the 4-bit selector nibble is
  //     always in [0, 16) so the source lane is always in-group;
  //     `bc=0` is the only case the emulation needs to support.
  //     Under SPE, `writeReg32`'s `emitUnderExec` already retains
  //     prior VDST values on EXEC-masked target lanes, covering the
  //     "target lane inactive" direction of `bc=0`.
  //   - `fi=0` and `bc=1` diverge from the above in ways the
  //     emulation does not model. Every GPT-OSS / softmax /
  //     bitmatrix disassembly we have examined uses `op_sel:[1, 0]`
  //     (fi=1, bc=0); refusing the other combinations keeps the
  //     classifier-gate's "no silent miscompile" invariant intact
  //     rather than emitting ds_bpermute with fi=0 semantics it
  //     does not provide.
  //
  // Future optimisation: on targets that DO support native
  // permlane16 (gfx10+), emit the intrinsic directly for lower
  // latency. Left as a profitability refinement -- correctness-first
  // lands the ds_bpermute emulation.
  case CanonicalOp::V_PERMLANE16_B32:
  case CanonicalOp::V_PERMLANEX16_B32: {
    const bool IsPermlaneX16 = (Sop == CanonicalOp::V_PERMLANEX16_B32);
    const bool Fi = (Op.srcMod(0) & SISrcMods::OP_SEL_0) != 0;
    const bool Bc = (Op.srcMod(1) & SISrcMods::OP_SEL_0) != 0;
    if (!Fi || Bc) {
      // Empirically the GPT-OSS / softmax / bitmatrix corpora emit
      // `op_sel:[1, 0]` exclusively (fi=1, bc=0). Refuse any other
      // encoding loudly so a future corpus kernel's extended
      // fi/bc use surfaces during classifier verification rather
      // than producing an approximation silently. Re-narrowing this
      // gate is the right place to extend the emulation.
      std::string Detail;
      raw_string_ostream Os(Detail);
      Os << "permlane16 / permlanex16 emulation supports only "
            "op_sel:[1,0] (fi=1, bc=0); saw fi="
         << (Fi ? 1 : 0) << ", bc=" << (Bc ? 1 : 0);
      return RaiseFailure::unsupportedInstructionForm(Di, "VALU", Detail);
    }
    Value *Src0 = Op.src(0);
    Value *Sel1 = Op.src(1);
    Value *Sel2 = Op.src(2);

    // Target-hardware lane id, wave-width-aware via emitLaneIdx, with
    // per-BB memoisation. Multiple permlane16 sites in the same BB
    // (e.g. butterfly reductions) reuse the single cached i32 instead
    // of re-emitting the mbcnt_lo / mbcnt_hi chain at each site --
    // LLVM's CSE would converge to the same end state, but the
    // pre-mem2reg IR stays smaller and lit-test-friendlier.
    Value *LaneId = Ctx.emitLaneIdx();

    // Group base (lane & ~0xF) and within-group index (lane & 0xF).
    Value *GroupBase =
        Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(~0xF), "pl_group");
    Value *Within = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(0xF), "pl_within");

    // Pick the right 32-bit selector word based on within's high bit.
    Value *IsHiHalf = Ctx.B.CreateICmpUGE(Within, Ctx.B.getInt32(8), "pl_hi");
    Value *SelWord = Ctx.B.CreateSelect(IsHiHalf, Sel2, Sel1, "pl_sel");

    // Extract the 4-bit nibble at position (within & 7) * 4.
    Value *WithinLo = Ctx.B.CreateAnd(Within, Ctx.B.getInt32(7), "pl_lo");
    Value *ShiftAmt = Ctx.B.CreateShl(WithinLo, Ctx.B.getInt32(2), "pl_shift");
    Value *Shifted = Ctx.B.CreateLShr(SelWord, ShiftAmt, "pl_shifted");
    Value *Nibble = Ctx.B.CreateAnd(Shifted, Ctx.B.getInt32(0xF), "pl_nibble");

    // For permlanex16, XOR the group base by 0x10 to swap adjacent groups.
    Value *SrcGroup =
        IsPermlaneX16
            ? Ctx.B.CreateXor(GroupBase, Ctx.B.getInt32(0x10), "plx_group")
            : GroupBase;
    Value *SrcLaneAbs = Ctx.B.CreateOr(SrcGroup, Nibble, "pl_src_lane");
    Value *ByteAddr = Ctx.B.CreateShl(SrcLaneAbs, Ctx.B.getInt32(2), "pl_addr");

    // Convergent: emit the bpermute outside any emitUnderExec diamond
    // so all hardware lanes participate. writeReg32 below wraps the
    // store for EXEC masking.
    Function *Bperm = Intrinsic::getOrInsertDeclaration(
        &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
    Value *Result =
        Ctx.B.CreateCall(Bperm, {ByteAddr, Src0},
                         IsPermlaneX16 ? "permlanex16_emu" : "permlane16_emu");
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_permlane64_b32 ----
  // KNOWN LIMITATION -- see the v_permlane64_b32 row in the
  // unrewritable table of hotswap/docs/wave-size-translation.md sec. 7:
  // no wave32 analogue, so
  // the Phase 1.4.5 classifier refuses this op in any cross-wave
  // lift (it is taxonomised as FullWaveRotate / unrewritable). The
  // same-lane fallback here only runs in same-wave (wave64 -> wave64)
  // translation, where a gfx1250 binary would not contain the op
  // anyway (gfx942 and earlier do not emit it). Keeping the stub
  // prevents a silent raise failure on the theoretical case.
  case CanonicalOp::V_PERMLANE64_B32: {
    if (Di.NumDefs >= 1 && Di.NumSrcs >= 1)
      Ctx.writeReg32(Op.dst(), Op.src(0));
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_readfirstlane_b32 sDST, vSRC ----
  // Broadcast the value of vSRC from the lowest-numbered active source lane
  // (or lane 0 if EXEC==0) to sDST.  Same-wave lowering can use the native
  // intrinsic directly.  Under cross-widening, however, native readfirstlane
  // would pick one lane for the entire target wave64 and collapse the two
  // source-wave halves together.  Emulate the source operation with
  // `ds_bpermute`: select the source-width slice of the modeled EXEC mask for
  // the current target lane's source-wave half, find that slice's first set
  // bit, and fetch the corresponding lane's VGPR.
  //
  // This is an explicit semantic translation, not a readfirstlane allow-list:
  // downstream scalar-looking uses now consume an already-broadcast
  // source-wave value that may differ between the two packed source waves.
  case CanonicalOp::V_READFIRSTLANE_B32: {
    Value *Src = Ctx.B.CreateZExtOrTrunc(Op.src(0), Ctx.I32Ty, "rfl_src");
    Value *Val = nullptr;
    if (Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
      Value *LaneId = Ctx.emitLaneIdx();
      uint32_t SourceMask = Ctx.Isa.WaveSize - 1;
      Value *GroupBase = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(~SourceMask),
                                         "rfl_source_wave_base");

      Value *Exec = Ctx.Regs.loadExec(Ctx.B);
      Value *ShiftAmt =
          Ctx.B.CreateZExtOrTrunc(GroupBase, Exec->getType(), "rfl_exec_shift");
      Value *SourceExecWide =
          Ctx.B.CreateLShr(Exec, ShiftAmt, "rfl_exec_at_srcwave");
      Value *SourceExec =
          Ctx.B.CreateTrunc(SourceExecWide, Ctx.I32Ty, "rfl_exec");
      Function *Cttz = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::cttz, {Ctx.I32Ty});
      Value *FirstSet = Ctx.B.CreateCall(
          Cttz, {SourceExec, ConstantInt::getFalse(Ctx.I1Ty)}, "rfl_first_set");
      Value *ExecIsZero =
          Ctx.B.CreateICmpEQ(SourceExec, Ctx.B.getInt32(0), "rfl_exec_is_zero");
      Value *SourceLane = Ctx.B.CreateSelect(ExecIsZero, Ctx.B.getInt32(0),
                                             FirstSet, "rfl_source_lane");
      Value *TargetLane =
          Ctx.B.CreateOr(GroupBase, SourceLane, "rfl_target_lane");
      Value *Addr =
          Ctx.B.CreateShl(TargetLane, Ctx.B.getInt32(2), "rfl_bperm_addr");
      Function *Bperm = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
      Val = Ctx.B.CreateCall(Bperm, {Addr, Src}, "readfirstlane_srcwave");
    } else {
      Function *Rfl = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_readfirstlane, {Ctx.I32Ty});
      Val = Ctx.B.CreateCall(Rfl, {Src}, "readfirstlane");
    }
    Ctx.writeReg32(Op.dst(), Val);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_writelane_b32 ----
  // Write `val` into lane `lane` of vDst. Cross-lane: cannot be
  // emulated via per-thread private scratch nor via a single scalar
  // SSA value. `llvm.amdgcn.writelane(val, lane, old)` lowers to the
  // hardware primitive; the intrinsic returns the new per-lane scalar
  // (either `val` when lane_id==lane, else `old`), so the VGPR's
  // SSA slot carries the correct value for whichever lane we are.
  //
  // First-write pattern: if writelane is the first assignment to
  // vDst, non-selected lanes legitimately hold whatever vDst
  // contained before (hardware semantics). `readReg32` on the
  // never-stored alloca returns LLVM `undef`, which is the right
  // "unobservable" encoding -- any downstream use of those lanes
  // before they are written is itself undefined on hardware.
  case CanonicalOp::V_WRITELANE_B32: {
    ParsedReg Dst = Op.dst();
    Value *Val = Op.src(0);
    Value *Lane = Op.src(1);
    Lane = Ctx.B.CreateZExtOrTrunc(Lane, Ctx.I32Ty, "wrlane_idx");
    Value *OldVal = Ctx.Regs.readReg32(Ctx.B, Dst);
    Value *NewVal = nullptr;
    // ThreadLoopProjection is the only projection that scopes lane ops to the
    // source wave inside the handler (`sourceWaveScopedLaneOps()`); the wider
    // ModuloReplicationProjection path leaves the native intrinsic here and
    // relies on the default-on post-raise `rewriteCrossLaneDivergent` pass to
    // rebase it symmetrically (with the SGPR-forced use-chain safety net), or
    // on the TLP re-raise fallback when that pass refuses.  See issue #146 and
    // wave-size-translation.md §5.6.3.
    if (Ctx.Projection.sourceWaveScopedLaneOps()) {
      Value *LaneId = Ctx.emitLaneIdx();
      Value *SourceLane = Ctx.B.CreateAnd(
          LaneId, Ctx.B.getInt32(Ctx.Isa.WaveSize - 1), "wrlane_source_lane");
      Value *WantedLane = Ctx.B.CreateAnd(
          Lane, Ctx.B.getInt32(Ctx.Isa.WaveSize - 1), "wrlane_wanted_lane");
      Value *IsTargetLane =
          Ctx.B.CreateICmpEQ(SourceLane, WantedLane, "wrlane_is_target_lane");
      NewVal =
          Ctx.B.CreateSelect(IsTargetLane, Val, OldVal, "writelane_srcwave");
    } else {
      Function *Wl = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_writelane, {Ctx.I32Ty});
      NewVal = Ctx.B.CreateCall(Wl, {Val, Lane, OldVal}, "writelane");
    }
    Ctx.writeReg32(Dst, NewVal);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_readlane_b32 sDST, vSRC, lane ----
  // Read a specific lane of vSRC into an SGPR. Reverse of writelane.
  case CanonicalOp::V_READLANE_B32: {
    ParsedReg SrcReg = Op.srcReg(0);
    Value *Lane = Op.src(1);
    Lane = Ctx.B.CreateZExtOrTrunc(Lane, Ctx.I32Ty, "rdlane_idx");
    Value *Src = Ctx.Regs.readReg32(Ctx.B, SrcReg);
    Value *Val = nullptr;
    // See the parallel note on V_WRITELANE_B32: the source-wave rebase here is
    // the ThreadLoopProjection path; under ModuloReplicationProjection the
    // native intrinsic is left for the default-on `rewriteCrossLaneDivergent`
    // pass (or the TLP re-raise fallback) to rebase.  Issue #146.
    if (Ctx.Projection.sourceWaveScopedLaneOps()) {
      Value *LaneId = Ctx.emitLaneIdx();
      uint32_t SourceMask = Ctx.Isa.WaveSize - 1;
      Value *GroupBase = Ctx.B.CreateAnd(LaneId, Ctx.B.getInt32(~SourceMask),
                                         "rdlane_source_wave_base");
      Value *SourceLane = Ctx.B.CreateAnd(Lane, Ctx.B.getInt32(SourceMask),
                                          "rdlane_source_lane");
      Value *TargetLane =
          Ctx.B.CreateOr(GroupBase, SourceLane, "rdlane_target_lane");
      Value *Addr =
          Ctx.B.CreateShl(TargetLane, Ctx.B.getInt32(2), "rdlane_bperm_addr");
      Function *Bperm = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_ds_bpermute);
      Val = Ctx.B.CreateCall(Bperm, {Addr, Src}, "readlane_srcwave");
    } else {
      Function *Rl = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_readlane, {Ctx.I32Ty});
      Val = Ctx.B.CreateCall(Rl, {Src, Lane}, "readlane");
    }
    Ctx.writeReg32(Op.dst(), Val);
    Hr.Handled = true;
    return Hr;
  }

  // ---- v_mbcnt_lo_u32_b32 / v_mbcnt_hi_u32_b32 ----
  // Count set bits in src0 below the current lane.  For same-wave lifts the
  // raw intrinsic is exact.  For wave32 source -> wave64 target, however,
  // raw target `mbcnt.lo` would return popcount(src0[0:31]) for target lanes
  // 32..63, while the source instruction's lane id restarts at 0 in the
  // second modeled source wave.  Recompute the source-wave-local low-half
  // count from `lane_id mod W_s` in that case.
  case CanonicalOp::V_MBCNT_LO_U32_B32: {
    Value *Result = nullptr;
    if (Ctx.Isa.isWave32() && Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
      Value *LaneId = Ctx.emitLaneIdx();
      Value *SourceLane = Ctx.B.CreateAnd(
          LaneId, Ctx.B.getInt32(Ctx.Isa.WaveSize - 1), "mbcnt_source_lane");
      Value *LaneBit =
          Ctx.B.CreateShl(Ctx.B.getInt32(1), SourceLane, "mbcnt_lane_bit");
      Value *BelowMask =
          Ctx.B.CreateSub(LaneBit, Ctx.B.getInt32(1), "mbcnt_below_mask");
      Value *SrcMask = Ctx.readOpSourceWaveMask32(Di, Op.srcIdx(0));
      Value *Masked = Ctx.B.CreateAnd(SrcMask, BelowMask, "mbcnt_masked");
      Function *Ctpop = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::ctpop, {Ctx.I32Ty});
      Result = Ctx.B.CreateAdd(Ctx.B.CreateCall(Ctpop, {Masked}, "mbcnt_pop"),
                               Op.src(1), "mbcnt_lo_srcwave");
    } else {
      Function *Mbcnt = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_mbcnt_lo, {});
      Result = Ctx.B.CreateCall(Mbcnt, {Op.src(0), Op.src(1)}, "mbcnt_lo");
    }
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }
  case CanonicalOp::V_MBCNT_HI_U32_B32: {
    // For wave32 source, mbcnt_hi is always a pass-through of src1: the
    // hi-half mask is `(1 << (lane_id - 32)) - 1`, which is 0 for every
    // source lane (0..31). When widened to wave64, the raw target
    // intrinsic would compute popcount(src0 & non_empty_mask) + src1 on
    // lanes 32..63, corrupting source-wave-1 results. Emit src1 directly
    // in the widening case.
    Value *Result = nullptr;
    if (Ctx.Isa.isWave32() && Ctx.TargetIsa.WaveSize > Ctx.Isa.WaveSize) {
      Result = Op.src(1);
    } else {
      Function *Mbcnt = Intrinsic::getOrInsertDeclaration(
          &Ctx.M, Intrinsic::amdgcn_mbcnt_hi, {});
      Result = Ctx.B.CreateCall(Mbcnt, {Op.src(0), Op.src(1)}, "mbcnt_hi");
    }
    Ctx.writeReg32(Op.dst(), Result);
    Hr.Handled = true;
    return Hr;
  }

  default:
    break;
  }
  return Hr;
}

} // namespace COMGR::hotswap
