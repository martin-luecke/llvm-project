//===-- handle_valu_output_mods.hpp - VOP3 output modifier guards --------===//
//
// Shared clamp/omod refusal for VALU handlers. Centralises operand reads and
// diagnostic wording so e32/e64-shared and VOP3-only opcodes do not drift.
//
//===----------------------------------------------------------------------===//

#ifndef AMD_COMGR_HOTSWAP_HANDLE_VALU_OUTPUT_MODS_HPP
#define AMD_COMGR_HOTSWAP_HANDLE_VALU_OUTPUT_MODS_HPP

#include "canonical_op.hpp"
#include "handlers.hpp"

#include "llvm/ADT/StringRef.h"

namespace transpiler {

/// Whether clamp/omod operands must exist in the MC operand table.
enum class VOP3OutputModPresence {
  /// e32 may omit both; e64/VOP3 forms must expose them when present.
  IfPresent,
  /// VOP3-only opcodes: both operands are required.
  Required,
};

/// Diagnostic shape for non-default or malformed output modifiers.
enum class VOP3OutputModDiag {
  /// Single combined message (small-op converts sharing e32/e64 CanonicalOps).
  Combined,
  /// Separate clamp vs omod messages (FP VALU ternary / clamp family).
  FpValuSplit,
  /// gfx12 VOP3 pseudo-scalar profile (hardware-intrinsic base lifts).
  PseudoScalar,
};

/// Refuse non-default VOP3 output modifiers (clamp / omod).
///
/// \p diagnosticName is printed in errors (mnemonic string or canonicalOpName).
bool requireDefaultVOP3OutputMods(const DecodedInst &di, HandlerResult &hr,
                                   llvm::StringRef diagnosticName,
                                   VOP3OutputModPresence presence,
                                   VOP3OutputModDiag diag);

inline bool requireDefaultOutputModsIfPresent(const DecodedInst &di,
                                              HandlerResult &hr) {
  return requireDefaultVOP3OutputMods(di, hr, canonicalOpName(di.canonOp),
                                    VOP3OutputModPresence::IfPresent,
                                    VOP3OutputModDiag::Combined);
}

inline bool requireDefaultPseudoScalarOutputMods(const DecodedInst &di,
                                                 HandlerResult &hr) {
  return requireDefaultVOP3OutputMods(di, hr, canonicalOpName(di.canonOp),
                                    VOP3OutputModPresence::Required,
                                    VOP3OutputModDiag::PseudoScalar);
}

/// VOP3-only FP VALU (e.g. min/max clamp). Used by handle_valu.cpp in PR #22+.
inline bool requireDefaultVOP3FpValuOutputMods(const DecodedInst &di,
                                               HandlerResult &hr,
                                               const char *opName) {
  return requireDefaultVOP3OutputMods(di, hr, opName,
                                    VOP3OutputModPresence::Required,
                                    VOP3OutputModDiag::FpValuSplit);
}

} // namespace transpiler

#endif // AMD_COMGR_HOTSWAP_HANDLE_VALU_OUTPUT_MODS_HPP
