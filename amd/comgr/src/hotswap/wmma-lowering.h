//===- wmma-lowering.h - Hotswap transpiler -------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_WMMA_LOWERING_H
#define HOTSWAP_TRANSPILER_WMMA_LOWERING_H

#include "llvm/Support/Error.h"

namespace llvm {
class Value;
} // namespace llvm

namespace COMGR::hotswap {

struct RaiseContext;

/// Lower a Wave32 v_wmma_f32_16x16x4_f32 (gfx1250 RDNA4 VOP3P opcode
/// 0x05D) to Wave64 mfma_f32_16x16x4f32 (gfx942 CDNA3) using
/// ds_bpermute lane redistribution.
///
/// This K=4 f32 variant is STRUCTURALLY DISTINCT from the K=32 / K=64
/// family covered by `emitWMMAtoMFMA` above:
///
///   * A/B fragment per Wave32 lane is `<2 x f32>` (2 VGPRs) -- not
///     the <16 x t> (16-bit element) or <8 x i32> (8-bit element)
///     8-VGPR fragments used by the K=32 / K=64 variants.
///   * Only ONE MFMA call per Wave32 group -- not 2 chained. The
///     source WMMA is already K=4, which exactly matches the target
///     `mfma_f32_16x16x4f32` K dimension; there is no K-tiling to do
///     inside a group.
///   * The MFMA A/B input is a single `float` per Wave64 lane -- not
///     `<4 x t>` (16-bit) or `i64` (8-bit).
///
/// The C/D fragment shape (<8 x f32> per Wave32 lane, Wave32 C-layout
/// equation `i = 8*floor(lane/16) + GPR`) is IDENTICAL to the K=32 /
/// K=64 variants, so the accumulator redistribution and the final
/// result collection reuse the same internal helpers as `emitWMMAtoMFMA`.
///
/// LAYOUT NOTE. The gfx1250 V_WMMA_F32_16x16x4_F32 per-lane A/B
/// (i, k) layout is extrapolated from the documented K=32 / K=64
/// pattern -- the AMD Matrix Instruction Calculator does not yet list
/// the K=4 f32 variant. We assume:
///
///   A/B input (<2 x f32> per lane):
///     i = lane % 16
///     k = 2*floor(lane/16) + GPR
///   Per-lane:
///     Lanes 0-15, GPR 0 -> k=0    GPR 1 -> k=1
///     Lanes 16-31, GPR 0 -> k=2   GPR 1 -> k=3
///
/// This is the unique natural extension of the K=32/K=64 layout
/// documented in wmma-lowering.cpp (lower half holds lower k-range,
/// upper half holds upper k-range; GPR indexes along k within a
/// lane-half). The layout is validated out-of-band by the hipBLASLt
/// Tensile `SS_SS_HA_Bias_SAV_UA` f32 GEMM kernels (macro-tile
/// `MT32x32x16`, WMMA shape `MI16x16x1`, wave32) -- see
/// a cross-target numerical comparison against the gfx1250 reference. A layout
/// mismatch would surface as a numerical regression there, not a
/// silent wrong answer.
///
/// \param a  WMMA source A fragment (<2 x f32> in Wave32)
/// \param b  WMMA source B fragment (<2 x f32> in Wave32)
/// \param c  WMMA accumulator fragment (<8 x f32> in Wave32)
/// \returns  `<8 x float>` -- result in Wave32 C-layout. Returns an error for
///           an unsupported source-wave projection.
llvm::Expected<llvm::Value *> emitWmmAtoMfmaF3216x16x4(RaiseContext &Ctx,
                                                       llvm::Value *A,
                                                       llvm::Value *B,
                                                       llvm::Value *C);

} // namespace COMGR::hotswap

#endif
