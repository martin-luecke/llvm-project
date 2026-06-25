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
// every gfx942 (FNUZ) fp8 hardware boundary the bytes are re-encoded.  gfx942
// fp8 E4M3 cannot represent OCP values in (240, 448]; those saturate to 240
// uniformly across every fp8 path on gfx942.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_FP8_CONVERT_H
#define HOTSWAP_TRANSPILER_FP8_CONVERT_H

#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace llvm {
class Value;
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

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_FP8_CONVERT_H
