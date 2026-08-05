//===- fp8-convert.h - Hotswap transpiler --------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// fp8/bf8 OCP <-> FNUZ re-encoding shared across the hotswap lowerings.
//
// fp8 has two incompatible numeric interpretations of the same byte:
//   * OCP  : E4M3FN (bias 7, max 448, NaN=S.1111.111, has -0, no Inf) and
//            E5M2 (bias 15, IEEE-style Inf/NaN, max 57344).  Used by gfx950
//            (CDNA4) and gfx12 / gfx1250 (RDNA).
//   * FNUZ : E4M3FNUZ (bias 8, max 240) and E5M2FNUZ (bias 16, max 57344);
//            no Inf, a single NaN encoding 0x80, no -0.  Used by gfx940 /
//            gfx941 / gfx942 (CDNA3).
//
// The raiser keeps in-register fp8 bytes in the SOURCE representation; at
// every gfx942 (FNUZ) fp8 hardware boundary the bytes are re-encoded.
//
// The byte re-encoders below define three policy classes on top of a plain
// round-half-to-even format conversion:
//   * NaN, and Inf (E5M2 only), map to the target's canonical NaN -- FNUZ
//     0x80, OCP 0x7F.  This is what APFloat does converting between the
//     matching semantics, neither target format having Inf.
//   * A finite magnitude above the target's max saturates, sign preserved.
//     This only arises for OCP E4M3 -> FNUZ E4M3, where OCP's (240, 448]
//     collapses onto 240; every other direction's range is a superset.
//   * FNUZ has no -0, so OCP -0 becomes +0.
//
// Saturation is a property of THESE converters, not of gfx942 fp8 generally:
// the f32 -> fp8 hardware encode (`v_cvt_pk_fp8_f32` et al.) yields NaN rather
// than a clamp for out-of-range inputs under the default MODE.FP16_OVFL=0.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_FP8_CONVERT_H
#define HOTSWAP_TRANSPILER_FP8_CONVERT_H

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace llvm {
class Value;
class Function;
template <typename FolderTy, typename InserterTy> class IRBuilder;
class ConstantFolder;
class IRBuilderDefaultInserter;
} // namespace llvm

namespace COMGR::hotswap {

struct ISAProfile;

/// Numeric interpretation of an fp8/bf8 byte on a given ISA.
enum class Fp8Format { None, OCP, FNUZ };

/// Classify how an ISA's fp8/bf8 hardware (MFMA operands, v_cvt_*_fp8/bf8)
/// interprets fp8 bytes.  FNUZ is CDNA3 (gfx940/941/942); every other
/// fp8-capable target (gfx950 CDNA4, gfx12 / gfx1250 RDNA) is OCP.
Fp8Format fp8FormatOf(const ISAProfile &P);

/// Data flow across an fp8/bf8 hardware boundary: SrcToTgt for hardware inputs
/// (MFMA/WMMA operands, decode inputs), TgtToSrc for hardware outputs (encode
/// results).
enum class Fp8Dir { SrcToTgt, TgtToSrc };

/// If \p Src and \p Tgt interpret fp8/bf8 bytes differently, return the
/// `ToFnuz` argument to pass to convertFp8Dword to re-encode in direction
/// \p Dir; otherwise nullopt (formats match, no re-encode needed).
std::optional<bool> fp8Reencode(const ISAProfile &Src, const ISAProfile &Tgt,
                                Fp8Dir Dir);

using HotswapIRBuilder =
    llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter>;

/// Per-byte-lane converters over a `<N x i32>` where each lane holds a byte
/// value 0..255; return a `<N x i32>` of re-encoded bytes.  Verified
/// exhaustively over all 256 byte values.
llvm::Value *convertOcpE4M3ToFnuz(HotswapIRBuilder &B, llvm::Value *Bytes);
llvm::Value *convertOcpE5M2ToFnuz(HotswapIRBuilder &B, llvm::Value *Bytes);
llvm::Value *convertFnuzE4M3ToOcp(HotswapIRBuilder &B, llvm::Value *Bytes);
llvm::Value *convertFnuzE5M2ToOcp(HotswapIRBuilder &B, llvm::Value *Bytes);

/// Re-encode a packed fp8/bf8 dword (4 bytes) through one of the byte-lane
/// converters above.  \p IsBf8 selects E5M2 vs E4M3; \p ToFnuz selects the
/// OCP->FNUZ vs FNUZ->OCP direction.
llvm::Value *convertFp8Dword(HotswapIRBuilder &B, llvm::Value *Dword,
                             bool IsBf8, bool ToFnuz);

/// Re-encode an array of packed fp8/bf8 dwords in place (see convertFp8Dword).
void convertFp8DwordsInPlace(HotswapIRBuilder &B,
                             llvm::SmallVectorImpl<llvm::Value *> &Dwords,
                             bool IsBf8, bool ToFnuz);

/// Decode one fp8/bf8 byte (\p Byte is an i32 holding 0..255) to f32, reading
/// it as \p Fmt.  Exact for all 256 inputs, including subnormals, Inf and NaN.
///
/// Used instead of the target's fp8 decode hardware when the source and target
/// formats differ.  byte -> f32 is a WIDENING conversion, so every source byte
/// has an exact f32 image; routing it through a byte re-encode plus the
/// target's decoder would clip the source's range for no reason (OCP E5M2 Inf,
/// OCP E4M3's (240, 448], and -0 all survive here).
///
/// The result is named `{fp8,bf8}_dec_{ocp,fnuz}`; lit fixtures match on that
/// to tell the conversion direction apart, so it is a test contract.
llvm::Value *decodeFp8ByteToF32(HotswapIRBuilder &B, llvm::Value *Byte,
                                bool IsBf8, Fp8Format Fmt);

/// Encode two f32 into two OCP fp8/bf8 bytes, packed into the low 16 bits.
///
/// \p CvtFn is the target's `cvt_pk_{fp8,bf8}_f32` (an FNUZ encoder).  OCP and
/// FNUZ share a mantissa width and differ by exactly one in exponent bias, so
/// the FNUZ encoding of x/2 IS the OCP encoding of x -- the target hardware
/// therefore does the round-half-to-even, and this only has to keep it inside
/// its own range.  Out-of-range magnitudes, NaN and signed zero are handled
/// explicitly, so MODE.FP16_OVFL (which makes the raw encoder return NaN
/// rather than clamp) never comes into play.
///
/// Only valid when the SOURCE format is OCP; the mirrored trick does not work
/// for an FNUZ source, whose top exponent has no OCP counterpart.
///
/// The result is named `pk_fp8_ocp`, which lit fixtures match on.
llvm::Value *encodeF32PairToOcpFp8(HotswapIRBuilder &B, llvm::Function *CvtFn,
                                   llvm::Value *S0, llvm::Value *S1,
                                   bool IsBf8);

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_FP8_CONVERT_H
