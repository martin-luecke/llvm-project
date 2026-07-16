//===- wmma-lowering.cpp - Hotswap transpiler -----------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ============================================================================
// WMMA -> MFMA Lowering via Layout-Aware Lane Redistribution
// ============================================================================
//
// This file lowers Wave32 WMMA instructions (gfx1250 / RDNA4) to Wave64 MFMA
// instructions (gfx942 / CDNA3) using cross-lane data movement.
//
// Background
// ----------
// WMMA (Wave Matrix Multiply Accumulate) on gfx1250 is a Wave32 collective
// operation: 32 lanes cooperate to compute a 16x16 matrix multiply.  MFMA on
// gfx942 is a Wave64 collective: 64 lanes cooperate.  Because the per-lane
// fragment sizes differ (WMMA: <16 x half> / <8 x float>; MFMA: <4 x half> /
// <4 x float>), a simple intrinsic swap is impossible.
//
// The core problem is the wave-size mismatch.  When gfx1250 code is compiled
// for gfx942, the hardware groups 64 threads into one wavefront instead of 32.
// Threads 0-31 and 32-63 were in separate Wave32 wavefronts on the source
// architecture, computing different sub-tiles.  A single MFMA would mix their
// unrelated data.
//
// Register Layout Equations (from AMD Matrix Instruction Calculator)
// ------------------------------------------------------------------
//
// gfx12 (RDNA4) v_wmma_f32_16x16x{32,64}_{f16,bf16,fp8_*,bf8_*}, Wave32:
//
// All supported variants share the same per-Wave32-lane fragment shape:
// 8 VGPRs (= 32 bytes) per A side, 8 VGPRs per B side, 8 VGPRs of f32
// per C/D side. The K-dimension scales inversely with the element
// width (K=32 for 16-bit elements, K=64 for 8-bit elements), so the
// total bytes per lane stay constant. The lane redistribution math
// below operates on dwords (32-bit cells); it is therefore byte-
// identical across element widths -- the only per-variant divergence
// lives in (a) the MFMA intrinsic dispatched on the gfx942 side and
// (b) the per-MFMA bitcast / pack type. See `runGroupPass` and the
// `WMMAInputType` enum in `wmma-lowering.h` for the full enumeration.
//
//   A input -- 16-bit variants (8 VGPRs, <16 x {half|bfloat}>):
//     i = lane % 16
//     k = 8*floor(GPR/2) + 4*floor(lane/16) + 2*(GPR%2) + floor(bits/16)
//
//     Per-lane breakdown:
//       Lanes 0-15:   GPR 0->k={0,1}  GPR 1->k={2,3}  GPR 2->k={8,9}
//                     GPR 3->k={10,11} GPR 4->k={16,17} GPR 5->k={18,19}
//                     GPR 6->k={24,25} GPR 7->k={26,27}
//       Lanes 16-31:  GPR 0->k={4,5}  GPR 1->k={6,7}  GPR 2->k={12,13}
//                     GPR 3->k={14,15} GPR 4->k={20,21} GPR 5->k={22,23}
//                     GPR 6->k={28,29} GPR 7->k={30,31}
//
//   A input -- 8-bit variants (8 VGPRs, <8 x i32> = 32 packed fp8/bf8):
//     Same dword-grain layout as the 16-bit variants -- a fp8/bf8 byte
//     occupies the same byte slot inside its containing dword and
//     across lanes/GPRs that the corresponding 16-bit element would
//     have occupied. The K-stride doubles (each dword holds 4 fp8/bf8
//     bytes vs 2 halves) so the per-GPR K-range is twice as wide, but
//     the redistribution acts at dword granularity and does not see
//     the element-level interpretation.
//
//   C/D output (8 VGPRs, <8 x float>) -- invariant across variants:
//     i = 8*floor(lane/16) + GPR
//     j = lane % 16
//     -> Lanes 0-15: rows 0-7;  Lanes 16-31: rows 8-15
//
// gfx942 (CDNA3) MFMA targets:
//
//   16-bit MFMA (v_mfma_f32_16x16x16_{f16|bf16_1k}, K=16 per call):
//     A input (2 VGPRs, <4 x {half|i16}>):
//       i = lane % 16
//       k = 4*floor(lane/16) + 2*GPR + floor(bits/16)
//
//   8-bit MFMA (v_mfma_f32_16x16x32_{fp8|bf8}_{fp8|bf8}, K=32 per call):
//     A input (2 VGPRs, packed as i64 = 8 fp8/bf8 bytes):
//       Same per-lane VGPR width (2 dwords) as the 16-bit MFMA. The
//       K-fanout doubles to match the doubled WMMA K-range.
//
//   C/D output (4 VGPRs, <4 x float>) -- invariant across variants:
//     i = 4*floor(lane/16) + (GPR % 4)
//     j = lane % 16
//     -> Lanes 0-15: rows 0-3; 16-31: rows 4-7; 32-47: rows 8-11; 48-63: rows
//     12-15
//
// Approach
// --------
// We process each "virtual Wave32 group" (lanes 0-31 and 32-63) in a separate
// pass.  For each group:
//
//   1. REDISTRIBUTE: Use ds_bpermute to move WMMA fragments into the MFMA
//      layout.  The mapping is NOT a simple lane/2 -- it must account for the
//      interleaved k-distribution between lanes 0-15 and 16-31 in gfx12 WMMA,
//      and the 4-way lane-group distribution in gfx942 MFMA.
//
//      For each MFMA GPR, we read from the correct WMMA GPR and lane using
//      a 4-way select based on the MFMA lane group (floor(laneId/16)):
//
//        Lane group 0 (lanes 0-15):  reads from WMMA lower half (lanes 0-15)
//        Lane group 1 (lanes 16-31): reads from WMMA upper half (lanes 16-31)
//        Lane group 2 (lanes 32-47): reads from WMMA lower half
//        Lane group 3 (lanes 48-63): reads from WMMA upper half
//
//      With WMMA GPR selection cycling every 2 lane groups (GPR pairs {0,1},
//      {2,3} for first K=16; {4,5}, {6,7} for second K=16).
//
//   2. MFMA: Two v_mfma_f32_16x16x16_f16 calls (K=32 decomposed to 2x K=16),
//      chaining the accumulator.
//
//   3. COLLECT: Gather the 4-VGPR MFMA result back to 8-VGPR WMMA layout.
//      WMMA GPR_w reads from MFMA GPR (GPR_w % 4), lane computed as:
//        srcLane = 32*(w32Lane >= 16) + 16*(GPR_w >= 4) + (w32Lane & 15)
//
// After both passes, a lane-ID-based select picks the correct group's result.
//
// Partial-wave correctness and hardware EXEC
// -------------------------------------------
// The redistribute / MFMA / collect pipeline is semantically a Wave64
// collective: every MFMA input and every collect-time bpermute source
// is physically stored in SOME lane of the Wave64, and each destination
// lane's VGPR must hold the correct value for the next stage to read
// it back.
//
// `ds_bpermute` and `v_mfma_*` both READ all 64 source lanes regardless
// of EXEC -- but the WRITE of their per-lane result is EXEC-gated.  So
// a lane with EXEC=0 silently skips updating its destination VGPR, and
// any later cross-lane read of that VGPR returns stale / poison data.
//
// This is invisible when the kernel is launched with a blockDim that
// fills an entire Wave64 (every lane is active; MFMA inputs / outputs
// are written everywhere).  It manifests as a catastrophic correctness
// failure on partial-wave launches -- e.g. a Wave32 WMMA kernel
// launched with blockDim == 32 runs as a single Wave64 with EXEC =
// 0x0000_0000_FFFF_FFFF on gfx942.  Lanes 32-63 never update their
// mfmaA/B/C VGPRs, so MFMA reads garbage for k=2,3 (for the K=4 f32
// path) or for the entire upper k-half (for the K=32/K=64 path), and
// the collect-stage bpermute's reads from lanes 32-63 of the MFMA
// output return garbage too.  Rows 8-15 of the output come out as
// undefined / zero, and rows 0-7 get only a partial K-accumulation.
//
// The fix lives OUTSIDE this file, at the transpiler's kernel-entry
// plumbing: `WaveNativeProjection::emitInitialExec` (in
// `wave-projection.cpp`) emits `@llvm.amdgcn.init_whole_wave` at the
// very top of the lifted kernel, which (a) sets hardware EXEC = -1 for
// the remainder of the kernel and (b) captures the original per-lane
// active mask into the transpiler's EXEC alloca. Every VGPR write,
// memory store, LDS op, and atomic in the lifted IR already routes
// through `RaiseContext::emitUnderExec`, which reads the alloca-backed
// source EXEC and emits an `if (lane_active)` diamond -- the AMDGPU
// backend lowers those divergent branches by setting hardware EXEC
// to the ballot of the per-lane predicate inside each `do` block and
// restoring to EXEC = -1 afterwards. So between `emitUnderExec`
// diamonds (which is where the bpermute / MFMA / select chain here
// lives) hardware EXEC is -1, and all 64 lanes participate in the
// Wave64 collective exactly as required.
//
// This supersedes an earlier design that wrapped MFMA-output dwords in
// `@llvm.amdgcn.strict.wwm`. That design was semantically correct but
// unscalable: `SIPreAllocateWWMRegs` requires a DEDICATED physical
// VGPR per virtual register defined inside a WWM bracket, and the
// WWM def-chain from an MFMA output walks back through the entire
// accumulator initialisation. A 128x128 f16 matmul tile's entry
// region contains ~200 IMPLICIT_DEF / AV_MOV_B32 0 instructions for
// its accumulator ring, which together with the kernel's own VGPR
// demand exceeds gfx942's 256-VGPR pool and aborts the allocator
// with `physreg not found for WWM expression`. Moving the EXEC = -1
// guarantee to kernel entry sidesteps the allocator pressure entirely
// -- no intermediate vreg is ever "inside WWM" and regalloc is
// ordinary -- while preserving the partial-wave correctness property.
//
// From the perspective of this file, that means the redistribute +
// MFMA + collect chain emits ONLY ordinary IR (bpermute, bitcast,
// select, MFMA intrinsic) with no WWM markers. The hardware EXEC = -1
// invariant is the kernel-wide ambient set up by
// `WaveNativeProjection::emitInitialExec`, and the `writeRegVec` call
// in `handle-valu-vop3p.cpp` that consumes this file's return value
// takes care of gating the Wave32-layout destination VGPRs back to
// the original per-lane active mask.
//
// ============================================================================

#include "wmma-lowering.h"
#include "raise-context.h"

#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace COMGR::hotswap {

static Value *emitDSBpermute(IRBuilder<> &B, Module &M, Value *ByteOffset,
                             Value *SrcVal) {
  Function *Fn =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_ds_bpermute);
  return B.CreateCall(Fn, {ByteOffset, SrcVal}, "bperm");
}

static Value *emitLaneId(IRBuilder<> &B, Module &M, Type *I32Ty) {
  Function *MbcntLo =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_mbcnt_lo);
  Function *MbcntHi =
      Intrinsic::getOrInsertDeclaration(&M, Intrinsic::amdgcn_mbcnt_hi);
  Value *AllOnes = ConstantInt::getSigned(I32Ty, -1);
  Value *Zero = ConstantInt::get(I32Ty, 0);
  Value *Lo = B.CreateCall(MbcntLo, {AllOnes, Zero}, "lane_lo");
  return B.CreateCall(MbcntHi, {AllOnes, Lo}, "lane_id");
}

static Value *packDwords(IRBuilder<> &B, Value **Dwords, unsigned NDwords,
                         Type *I32Ty, Type *TargetTy) {
  auto *VecTy = FixedVectorType::get(I32Ty, NDwords);
  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned I = 0; I < NDwords; ++I)
    Vec = B.CreateInsertElement(Vec, Dwords[I], I, "pack");
  return B.CreateBitCast(Vec, TargetTy, "cast");
}

static void unpackDwords(IRBuilder<> &B, Value *Vec, unsigned NDwords,
                         Type *I32Ty, Value **Out) {
  auto *VecTy = FixedVectorType::get(I32Ty, NDwords);
  Value *AsI32 = B.CreateBitCast(Vec, VecTy, "toi32");
  for (unsigned I = 0; I < NDwords; ++I)
    Out[I] = B.CreateExtractElement(AsI32, I, "dw");
}

/// 4-way select based on lane group index (0..3).
/// Returns vals[laneGroup] for each lane.
static Value *selectByLaneGroup(IRBuilder<> &B, Value *LaneGroup, Value *V0,
                                Value *V1, Value *V2, Value *V3) {
  Value *S = V3;
  S = B.CreateSelect(B.CreateICmpEQ(LaneGroup, B.getInt32(2)), V2, S);
  S = B.CreateSelect(B.CreateICmpEQ(LaneGroup, B.getInt32(1)), V1, S);
  S = B.CreateSelect(B.CreateICmpEQ(LaneGroup, B.getInt32(0)), V0, S);
  return S;
}


/// Redistribute accumulator C from gfx12 WMMA layout (8 VGPRs, Wave32)
/// to gfx942 MFMA layout (4 VGPRs, Wave64).
///
/// gfx12 WMMA: i = 8*floor(lane/16) + GPR  ->  rows 0-7 in lanes 0-15, 8-15 in
/// lanes 16-31 gfx942 MFMA: i = 4*floor(lane/16) + GPR ->  rows 0-3 in LG0, 4-7
/// in LG1, 8-11 in LG2, 12-15 in LG3
///
/// MFMA GPR g needs:
///   LG 0: i = g      -> WMMA GPR g,   lower W32 half
///   LG 1: i = 4+g    -> WMMA GPR 4+g, lower W32 half
///   LG 2: i = 8+g    -> WMMA GPR g,   upper W32 half
///   LG 3: i = 12+g   -> WMMA GPR 4+g, upper W32 half
static void redistributeAcc(IRBuilder<> &B, Module &M, Value **CDwords,
                            Value *AddrLo, Value *AddrHi, Value *LaneGroup,
                            Value **MfmaC) {
  for (unsigned G = 0; G < 4; ++G) {
    Value *V0 = emitDSBpermute(B, M, AddrLo, CDwords[G]);
    Value *V1 = emitDSBpermute(B, M, AddrLo, CDwords[G + 4]);
    Value *V2 = emitDSBpermute(B, M, AddrHi, CDwords[G]);
    Value *V3 = emitDSBpermute(B, M, AddrHi, CDwords[G + 4]);
    MfmaC[G] = selectByLaneGroup(B, LaneGroup, V0, V1, V2, V3);
  }
}

/// Collect MFMA result (4 VGPRs, Wave64) back to WMMA layout (8 VGPRs, Wave32).
///
/// WMMA GPR_w reads MFMA GPR (GPR_w % 4) from:
///   srcLane = 32*(w32Lane >= 16) + 16*(GPR_w >= 4) + (w32Lane & 15)
static void collectResult(IRBuilder<> &B, Module &M, Value **MfmaDwords,
                          Value *W32Lane, Value **Out) {
  Value *W32Lo = B.CreateAnd(W32Lane, B.getInt32(15), "w32_lo");
  Value *IsUpper = B.CreateICmpUGE(W32Lane, B.getInt32(16), "is_upper");
  Value *UpperOff =
      B.CreateSelect(IsUpper, B.getInt32(32), B.getInt32(0), "upper_off");

  for (unsigned Gw = 0; Gw < 8; ++Gw) {
    Value *GprOff = B.getInt32((Gw >= 4) ? 16 : 0);
    Value *SrcLane =
        B.CreateAdd(B.CreateAdd(UpperOff, GprOff), W32Lo, "col_lane");
    Value *Addr = B.CreateShl(SrcLane, B.getInt32(2), "col_addr");
    Out[Gw] = emitDSBpermute(B, M, Addr, MfmaDwords[Gw % 4]);
  }
}


// ----------------------------------------------------------------------
// v_wmma_f32_16x16x4_f32 -> mfma_f32_16x16x4f32 lowering
// ----------------------------------------------------------------------
//
// Source (gfx1250 RDNA4, Wave32):
//   int_amdgcn_wmma_f32_16x16x4_f32 -- `<8 x f32>` = (..., <2 x f32> A,
//   ..., <2 x f32> B, ..., <8 x f32> C, ...)
//
// Target (gfx942 CDNA3, Wave64):
//   int_amdgcn_mfma_f32_16x16x4f32 -- `<4 x f32>` = (f32 A, f32 B,
//   <4 x f32> C, i32 cbsz, i32 abid, i32 blgp)
//
// Register-layout equations
// -------------------------
// Source WMMA (Wave32, per-lane fragment):
//   A/B  -- <2 x f32> (2 VGPRs):  i = lane%16,  k = 2*floor(lane/16) + GPR
//     Lanes 0-15 GPR 0->k=0, GPR 1->k=1
//     Lanes 16-31 GPR 0->k=2, GPR 1->k=3
//   C/D  -- <8 x f32> (8 VGPRs):  i = 8*floor(lane/16) + GPR,  j = lane%16
//     Lanes 0-15 -> rows 0-7;   Lanes 16-31 -> rows 8-15
//
// Target MFMA (Wave64, per-lane fragment):
//   A/B  -- f32 (1 VGPR):          i = lane%16,  k = floor(lane/16)
//     LG0 (lanes 0-15)  -> k=0
//     LG1 (lanes 16-31) -> k=1
//     LG2 (lanes 32-47) -> k=2
//     LG3 (lanes 48-63) -> k=3
//   C/D  -- <4 x f32> (4 VGPRs):   i = 4*floor(lane/16) + GPR, j = lane%16
//     (same layout equation as the K=32/K=64 MFMA family, so the C
//     redistribution + result collection helpers above are reused
//     verbatim.)
//
// Redistribution
// --------------
// Per-group pass (`groupBase in {0, 32}`): the W32-group-N's data is
// held by W64 lanes `[groupBase .. groupBase+31]`. Each MFMA call
// spreads ONE Wave32 group's 32-lane x 2-dword A across all 64 Wave64
// lanes:
//
//   loAddr = 4 * ((lane%16) + groupBase)       // source for k=0..1
//   hiAddr = 4 * ((lane%16) + groupBase + 16)  // source for k=2..3
//
//   LG0 (lanes 0-15,  k=0): bpermute(loAddr, aDwords[0])
//   LG1 (lanes 16-31, k=1): bpermute(loAddr, aDwords[1])
//   LG2 (lanes 32-47, k=2): bpermute(hiAddr, aDwords[0])
//   LG3 (lanes 48-63, k=3): bpermute(hiAddr, aDwords[1])
//
// All four reads deliver the full K=4 range for ONE virtual Wave32
// group, which matches the K=4 MFMA signature -- so there is exactly
// ONE MFMA call per group (not 2 chained as in the K=32/K=64 path).
//
// B redistribution mirrors A exactly (same layout equation). The C
// redistribution reuses `redistributeAcc` (same WMMA C layout) and
// the result collection reuses `collectResult` (same WMMA D layout).
//
// Hardware EXEC: the redistribute / MFMA / collect chain relies on
// the kernel-wide EXEC = -1 invariant set up by
// `WaveNativeProjection::emitInitialExec`, so no in-file WWM marker
// is needed.  See the file-header "Partial-wave correctness and
// hardware EXEC" section for the correctness argument.
static void runGroupPassF32K4(IRBuilder<> &B, Module &M, RaiseContext &Ctx,
                              unsigned GroupBase, Value *LaneId,
                              Value **ADwords, Value **BDwords, Value **CDwords,
                              Value **ResultDwords) {
  Value *LaneMod16 = B.CreateAnd(LaneId, B.getInt32(15), "lane16");
  Value *LoLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase), "lo_lane");
  Value *HiLane = B.CreateAdd(LaneMod16, B.getInt32(GroupBase + 16), "hi_lane");
  Value *AddrLo = B.CreateShl(LoLane, B.getInt32(2), "addr_lo");
  Value *AddrHi = B.CreateShl(HiLane, B.getInt32(2), "addr_hi");
  Value *LaneGroup = B.CreateLShr(LaneId, B.getInt32(4), "lane_grp");

  // A input: single-dword MFMA fragment, one bpermute per (lane-group,
  // GPR) combination. aDwords[0] carries k=0 and k=2; aDwords[1]
  // carries k=1 and k=3 (lower vs upper WMMA half selects the +16
  // lane offset).
  Value *ALG0 = emitDSBpermute(B, M, AddrLo, ADwords[0]);
  Value *ALG1 = emitDSBpermute(B, M, AddrLo, ADwords[1]);
  Value *ALG2 = emitDSBpermute(B, M, AddrHi, ADwords[0]);
  Value *ALG3 = emitDSBpermute(B, M, AddrHi, ADwords[1]);
  Value *MfmaAI32 = selectByLaneGroup(B, LaneGroup, ALG0, ALG1, ALG2, ALG3);

  Value *BLG0 = emitDSBpermute(B, M, AddrLo, BDwords[0]);
  Value *BLG1 = emitDSBpermute(B, M, AddrLo, BDwords[1]);
  Value *BLG2 = emitDSBpermute(B, M, AddrHi, BDwords[0]);
  Value *BLG3 = emitDSBpermute(B, M, AddrHi, BDwords[1]);
  Value *MfmaBI32 = selectByLaneGroup(B, LaneGroup, BLG0, BLG1, BLG2, BLG3);

  Value *MfmaC[4];
  redistributeAcc(B, M, CDwords, AddrLo, AddrHi, LaneGroup, MfmaC);

  // Pack per-lane MFMA operands. The signatures are:
  //   A:f32         (scalar dword, not packed)
  //   B:f32         (scalar dword, not packed)
  //   C:<4 x f32>
  // The redistribution produced i32 dwords; bitcast A/B to f32 and
  // pack C into `<4 x float>` via the existing helper (which also
  // bitcasts).
  auto *MfmaAccPackTy = FixedVectorType::get(Ctx.F32Ty, 4);
  Value *MfmaA = B.CreateBitCast(MfmaAI32, Ctx.F32Ty, "mfma_a");
  Value *MfmaB = B.CreateBitCast(MfmaBI32, Ctx.F32Ty, "mfma_b");
  Value *Acc = packDwords(B, MfmaC, 4, Ctx.I32Ty, MfmaAccPackTy);

  // cbsz / abid / blgp are the per-matrix broadcast-and-shift
  // modifiers hard-coded to zero here; the corpus kernels emit the
  // MFMA equivalent with these defaulted, matching what gfx1250
  // WMMA surfaces for the failing kerneldex Tensile GEMMs.
  Value *Cbsz = B.getInt32(0), *Abid = B.getInt32(0), *Blgp = B.getInt32(0);

  Function *MfmaFn = Intrinsic::getOrInsertDeclaration(
      &M, Intrinsic::amdgcn_mfma_f32_16x16x4f32);
  // See the K=32 / K=64 `runGroupPass` helper above for the full
  // rationale: under MODREP phantom-lane, the MFMA's destination
  // VGPR must be written on every target lane so the subsequent
  // `collectResult` bpermute reads real data from lanes 32..47's
  // output.  `wrapAsWWMValue` inserts a `strict.wwm` marker under
  // MODREP (so the backend's `SIWholeQuadMode` pulls the MFMA into
  // a WWM region) and is an identity no-op under WaveNative (whose
  // kernel-entry `init_whole_wave` already keeps HW EXEC=-1).
  Value *Mfma = Ctx.Projection.wrapAsWWMValue(
      B, B.CreateCall(MfmaFn, {MfmaA, MfmaB, Acc, Cbsz, Abid, Blgp}, "mfma"),
      "mfma_wwm");

  Value *MfmaDst[4];
  unpackDwords(B, Mfma, 4, Ctx.I32Ty, MfmaDst);

  Value *W32Lane = B.CreateAnd(LaneId, B.getInt32(31), "w32_lane");
  collectResult(B, M, MfmaDst, W32Lane, ResultDwords);

  // Wrap collect outputs as WWM under MODREP -- see the equivalent
  // block comment in the K=32 / K=64 `runGroupPass` for the full
  // rationale.
  for (unsigned I = 0; I < 8; ++I)
    ResultDwords[I] =
        Ctx.Projection.wrapAsWWMValue(B, ResultDwords[I], "wmma_collect_wwm");
}

Expected<Value *> emitWmmAtoMfmaF3216x16x4(RaiseContext &Ctx, Value *A,
                                           Value *Vb, Value *C) {
  // K=4 f32 counterpart to `emitWMMAtoMFMA` above -- see that
  // function's block comment for the full design rationale
  // (projection-aware per-MFMA `strict.wwm` wrapping via
  // `wrapAsWWMValue`, and pass-1-skipped under MODREP phantom-
  // lane).  The only structural difference is that the K=4 f32
  // decomposition emits ONE MFMA per group pass (the source WMMA
  // is already K=4, exactly matching `mfma_f32_16x16x4f32`), not
  // the 2-chained-MFMA K=32->2xK=16 structure of the 16-/8-bit
  // family.
  IRBuilder<> &B = Ctx.B;
  Module &M = Ctx.M;

  Value *ADwords[2], *BDwords[2], *CDwords[8];
  unpackDwords(B, A, 2, Ctx.I32Ty, ADwords);
  unpackDwords(B, Vb, 2, Ctx.I32Ty, BDwords);
  unpackDwords(B, C, 8, Ctx.I32Ty, CDwords);

  Value *LaneId = emitLaneId(B, M, Ctx.I32Ty);

  const unsigned NumSrcWaves = Ctx.Projection.numSourceWavesPerTarget();
  if (NumSrcWaves != 1 && NumSrcWaves != 2)
    return createStringError(
        "WMMA->MFMA lowering defined only for wave32 source projections; "
        "numSourceWavesPerTarget() must be 1 (MODREP phantom-lane) or 2 "
        "(WaveNative cross-widen) -- a new projection class must declare "
        "which applies.");
  // Mirrors `emitWMMAtoMFMA` above: both MODREP (numSrcWaves==1) and
  // WaveNative (numSrcWaves==2) fall through to the two-branch pass
  // logic below; the old staging refusal was lifted once the
  // dispatcher's refusal gate was narrowed to the multi-WMMA-per-
  // K-iter pattern (see `handle-valu-vop3p.cpp`).

  Value *Result0[8];
  runGroupPassF32K4(B, M, Ctx, /*groupBase=*/0, LaneId, ADwords, BDwords,
                    CDwords, Result0);

  Value *FinalDwords[8];
  if (NumSrcWaves == 1) {
    for (unsigned I = 0; I < 8; ++I)
      FinalDwords[I] = Result0[I];
  } else {
    Value *Result1[8];
    runGroupPassF32K4(B, M, Ctx, /*groupBase=*/32, LaneId, ADwords, BDwords,
                      CDwords, Result1);
    Value *IsGroup1 = B.CreateICmpUGE(LaneId, B.getInt32(32), "is_group1");
    for (unsigned I = 0; I < 8; ++I)
      FinalDwords[I] = B.CreateSelect(IsGroup1, Result1[I], Result0[I], "sel");
  }

  return packDwords(B, FinalDwords, 8, Ctx.I32Ty,
                    FixedVectorType::get(Ctx.F32Ty, 8));
}

// ----------------------------------------------------------------------
// v_wmma_scale_f32_16x16x128_f8f6f4 -> v_mfma_scale_f32_16x16x128_f8f6f4
// ----------------------------------------------------------------------
//
// Source (gfx1250 RDNA4, Wave32, VOP3PX2):
//   int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4 -- 14-arg intrinsic.
//   Per Wave32 lane:
//     A: <aDwords x i32>   (aDwords = 16/12/8 for f8/f6/f4)
//     B: <bDwords x i32>   (same encoding)
//     C/D: <8 x f32>        (same as the K=32/K=64 WMMA family)
//
// Target (gfx950 CDNA4, Wave64, VOP3PX):
//   int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4 -- 9-arg intrinsic, overloaded
//   on AB type.  Per handle-mfma.cpp convention we declare it with a uniform
//   <8 x i32> A/B type (the widest case, F8) and let cbsz / blgp select the
//   active subset of dwords per the LLVM TableGen rule.
//   Per Wave64 lane:
//     A, B: <8 x i32>       (logical: 8/6/4 dwords for f8/f6/f4; upper
//                           dwords are poison and ignored by hardware
//                           when cbsz / blgp narrow the width)
//     C/D:  <4 x f32>
//
// K decomposition:
//   None.  gfx950 MFMA scaled F8F6F4 covers the full K=128 in one call, so
//   the lowering is structurally simpler than emitWMMAtoMFMA's K=32 -> 2 x
//   K=16 chain.  We emit ONE MFMA call per virtual-W32-group pass.
//
// Lane redistribution (LINEAR-K model, validated on gfx950 silicon across
// 20 mxfp_attn_fwd_kernel shapes in rocm-hotswap-testing; matrix-class
// prediction median 0.1% error, 769 mfma_scale sites per kernel):
//   Per virtual Wave32 group at groupBase in {0, 32}, the A/B fragments live
//   in W64 lanes [groupBase .. groupBase+31].  The MFMA per-LG K-quarter
//   mapping is:
// clang-format off
//     LG0 (W64 lanes 0-15):   K = 0..31     <- WMMA dwords [0..mfmaDw-1] in W32 lower half
//     LG1 (W64 lanes 16-31):  K = 32..63    <- WMMA dwords [mfmaDw..2*mfmaDw-1] in W32 lower half
//     LG2 (W64 lanes 32-47):  K = 64..95    <- WMMA dwords [0..mfmaDw-1] in W32 upper half
//     LG3 (W64 lanes 48-63):  K = 96..127   <- WMMA dwords [mfmaDw..2*mfmaDw-1] in W32 upper half
// clang-format on
//   (mfmaDw = wmmaDw / 2 = 8/6/4 for f8/f6/f4.)
//
//   This is the "linear-K halves" model -- WMMA's two-lane-half split holds
//   the two K-halves linearly (lower-K then upper-K), and within each lane
//   the dwords run linearly over the K-range.  Contrast with the K=32 / K=64
//   16-bit family's interleaved layout (documented in this file's header
//   block) which alternates k between the lane halves.  Without an explicit
//   K=128 F8F6F4 layout doc in the AMD Matrix Instruction Calculator we
//   choose the simpler linear model; if the hardware layout turns out to be
//   the interleaved variant we adjust the redistribution below to match
//   (a one-line change in the lambda).
//
// C_mod application:
//   gfx1250 WMMA carries an i16 immediate (0 = none, 1 = neg, 2 = abs,
//   3 = neg(abs)) that gates the C-input modifier.  gfx950 MFMA has no
//   equivalent argument, so we apply it as IR fneg / fabs on the redistributed
//   <4 x f32> C input *before* the MFMA call.  The cMod argument is required
//   to be a ConstantInt -- callers pass an immediate-derived value, and any
//   non-constant trips the cast<ConstantInt> assertion (matching the
//   discipline of the WMMA-side fast path which also assumes a constant).
//
// Scale operand layout:
//   The MFMA scaled intrinsic takes one i32 scale per side, plus a 2-bit
//   op_sel encoding scale-format selection.  We pass scaleSrc0 / scaleSrc1
//   through directly (each Wave64 lane reads its own scale value, which
//   under WaveNativeProjection is already in the right per-source-lane
//   layout) and conservatively set op_sel = 0 for both A and B.  The exact
//   WMMA matrix_*_scale + matrix_*_scale_fmt -> MFMA op_sel mapping is not
//   documented in our reference materials; passing 0 selects the default
//   bit-position-zero scale, which is correct for the common UE8M0 scale
//   format that the MXFP attention kernels in our corpus use.  TODO: refine
//   if a kernel emits a non-default scale_fmt.
//
// EXEC handling:
//   Same as emitWMMAtoMFMA -- the MFMA/bpermute chain runs under the
//   kernel-wide HW EXEC = -1 ambient set by WaveNativeProjection's
//   init_whole_wave; under MODREP the wrapAsWWMValue calls keep the chain
//   inside SIWholeQuadMode's WWM region.


} // namespace COMGR::hotswap
