//===- raise_failure.h - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_H
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace COMGR::hotswap {

// Structured reason for a raise failure. Lives in its own header so the
// handler layer (`raise-context.h`) can depend on failure values
// without pulling in `RaiseResult` and the rest of the top-level
// `raiser.h` interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  // Caller-supplied input rejected before the MC stack is constructed,
  // e.g. an empty or non-AMDGPU source ISA string. `Detail` carries the
  // offending input string.
  BadInput,
  // Internal contract violation: a caller reached a failure return path without
  // the structured failure that should explain it. This is a Hotswap bug, not a
  // property of the source kernel.
  InternalError,
  // Main loop: no handler matched on TSFlags, or every matching handler
  // returned unhandled without setting a more specific failure. The
  // `mnemonic` / `format` / `offset` triple locates the instruction.
  UnsupportedOpcode,
  // A handler matched on CanonicalOp but the specific operand shape /
  // encoding variant it saw is not yet modelled. Today's format-
  // specific failure sites (handle_valu, handle_flat, handle_mubuf,
  // handle_mfma, handle_vopd) all use this category. `detail` carries
  // shape-specific context when available.
  UnsupportedInstructionForm,
  // A source metadata hidden-argument byte was identified, but Hotswap has no
  // explicit synthesis for that `.value_kind` yet. This is narrower than
  // `UnsupportedInstructionForm`: the instruction and SMEM hidden-arg path are
  // both recognized; only that source hidden argument kind is unsupported.
  UnsupportedSourceHiddenArg,
  // An EXEC-writing instruction whose CanonicalOp is not marked as
  // routing EXEC through storeExec.
  SPEUnsafeExecWriter,
  // `TargetRegistry::createTargetMachine` returned null.
  TargetMachineCreationFailed,
  // `verifyModule` rejected the emitted IR.
  IRVerificationFailed,
  // Decoded control flow targets outside the selected kernel symbol extent, or
  // the raiser could not decode an in-extent target required by static CFG
  // recovery. The selected kernel boundary is part of the source object
  // contract; crossing it would inspect/lift neighboring symbols.
  KernelBoundaryViolation,
  // A helper/device-library bitcode link step failed before verification. This
  // is distinct from verifier failure: the module is intentionally incomplete
  // until the embedded helper or device-library body is linked and inlined.
  DeviceLibraryLinkFailed,
  // Wave-size-obstruction refusals, one reason per refusal decision so
  // diagnostics can bucket failures without parsing the failure text.
  CrossWaveLaneIdLeak,
  CrossWaveUnrewritableShuffle,
  CrossWaveShuffleRewritePending,
  CrossWaveReplicaRace,
  CrossWaveLanePredicatedExec,
  // workitem.id.x() feeds a lane-position-scoped icmp that gates a side
  // effect, and the chain was not AND-masked to the source wave width.
  CrossWavePredicateChain,
  // `HSA_HOTSWAP_STRICT=1`-only refusal: a handler recognised the
  // CanonicalOp and would have lifted it under the warn-and-continue
  // policy, but strict mode requires an "unsupported, may silently
  // miscompile" verdict instead.
  StrictUnsafeLowering,
  // The kernel descriptor could not be read from .rodata via the
  // `<name>.kd` symbol, so UserSgprLayout cannot be derived and the
  // lift is refused.
  MissingKernelDescriptor,
  // The KD's raw USER_SGPR_COUNT field disagrees with the layout implied
  // by kernel_code_properties plus kernarg_preload for the source ISA.
  UserSgprLayoutMismatch,
  // The source code object declares non-disabled workgroup cluster
  // dimensions. TTMP6 then carries per-cluster workgroup state that the
  // HotSwap ABI model does not reconstruct.
  UnsupportedSourceClusterDims,
};

// Human-readable name for a `RaiseFailureReason`. Stable enough for
// diagnostics and tests to bucket on.
const char *reasonString(RaiseFailureReason R);

// RaiseFailure is both the structured failure value used throughout the
// handler layer and an `llvm::ErrorInfo`, so it can be carried directly as the
// payload of a `llvm::Error` / `Expected` failure (multiple failures are
// combined with `joinErrors`). A default-constructed value has
// `Reason == None` and does not represent a real failure.
struct RaiseFailure : public llvm::ErrorInfo<RaiseFailure> {
  static char ID;

  RaiseFailureReason Reason = RaiseFailureReason::None;
  // Offending instruction mnemonic (e.g. `global_store_dwordx4`).
  std::string Mnemonic;
  // Encoding-format category (e.g. `VALU`, `FLAT`, `MUBUF`) -- stable
  // bucketing key for the batch / corpus test summaries. For non-
  // decode-level failures (e.g. `TargetMachineCreationFailed`) this
  // is the `reasonString` of `Reason`.
  std::string Format;
  // Byte offset inside the disassembled text section, in host order.
  // Zero for failures not tied to a specific instruction.
  uint64_t Offset = 0;
  // Optional human-readable context; may include shape hints,
  // attempted rewrites, etc.
  std::string Detail;

  RaiseFailure() = default;

  // True when this value describes a real failure. A default-constructed
  // value (Reason == None) has not failed.
  bool hasFailed() const { return Reason != RaiseFailureReason::None; }
  RaiseFailure(RaiseFailureReason Reason, std::string Mnemonic,
               std::string Format, uint64_t Offset, std::string Detail)
      : Reason(Reason), Mnemonic(std::move(Mnemonic)),
        Format(std::move(Format)), Offset(Offset), Detail(std::move(Detail)) {}

  void log(llvm::raw_ostream &OS) const override;

  std::error_code convertToErrorCode() const override {
    return llvm::inconvertibleErrorCode();
  }

  // Handler recognised the CanonicalOp but refused the specific instruction
  // form, operand profile, or target capability. `Mnemonic` and `Offset`
  // identify the refused instruction.
  static llvm::Error unsupportedInstructionForm(llvm::StringRef Mnemonic,
                                                uint64_t Offset,
                                                llvm::StringRef Format,
                                                const llvm::Twine &Detail = {});

  // Handler identified a source metadata hidden argument, but no explicit
  // source-side synthesis exists for that .value_kind yet.
  static llvm::Error unsupportedSourceHiddenArg(llvm::StringRef Mnemonic,
                                                uint64_t Offset,
                                                llvm::StringRef Format,
                                                llvm::StringRef Detail);

  // Main loop: no handler claimed the CanonicalOp (either no TSFlags match
  // or every matching handler returned `handled=false` without setting a more
  // specific failure). `Format` is the human-readable encoding label.
  static llvm::Error unsupportedOpcode(llvm::StringRef Mnemonic,
                                       uint64_t Offset, llvm::StringRef Format);

  // An EXEC-writing instruction whose CanonicalOp is not marked as
  // routing EXEC through storeExec.
  static llvm::Error speUnsafeExecWriter(llvm::StringRef Mnemonic,
                                         uint64_t Offset,
                                         const llvm::Twine &Detail);

  // `TargetRegistry::createTargetMachine` returned null.
  static llvm::Error targetMachineCreationFailed();

  // Internal invariant violation surfaced as a structured failure so callers do
  // not misclassify it as an unsupported source instruction.
  static llvm::Error internalFailure(const llvm::Twine &Detail);

  // Caller-supplied input rejected before the MC stack is built (e.g. an
  // empty or non-AMDGPU source ISA string).
  static llvm::Error badInput(const llvm::Twine &Detail);

  // `verifyModule` rejected the emitted IR. `Err` carries the verifier's
  // diagnostic text for the `Detail` field.
  static llvm::Error irVerificationFailed(llvm::StringRef Err);

  // Kernel-symbol boundary check failed during CFG recovery.
  static llvm::Error kernelBoundaryViolation(llvm::StringRef KernelName,
                                             uint64_t TargetOffset,
                                             const llvm::Twine &Detail);

  // Embedded helper/device-library linking failed before verification.
  // `kernelName` and `detail` preserve attribution for proof logs without
  // mis-bucketing the failure as an LLVM verifier rejection.
  static llvm::Error deviceLibraryLinkFailed(llvm::StringRef KernelName,
                                             const llvm::Twine &Detail);

  // Wave-size-obstruction refusal. `Mnemonic` and `Offset` locate the
  // refused instruction; `KindDetail` carries human-readable operand-level
  // context (e.g. "operand value N >= wave width M").
  static llvm::Error crossWaveLaneIdLeak(llvm::StringRef Mnemonic,
                                         uint64_t Offset,
                                         const llvm::Twine &KindDetail);

  static llvm::Error
  crossWaveUnrewritableShuffle(llvm::StringRef Mnemonic, uint64_t Offset,
                               const llvm::Twine &KindDetail);

  static llvm::Error
  crossWaveShuffleRewritePending(llvm::StringRef Mnemonic, uint64_t Offset,
                                 const llvm::Twine &KindDetail);

  static llvm::Error crossWaveReplicaRace(llvm::StringRef Mnemonic,
                                          uint64_t Offset,
                                          const llvm::Twine &KindDetail);

  static llvm::Error crossWaveLanePredicatedExec(llvm::StringRef Mnemonic,
                                                 uint64_t Offset,
                                                 const llvm::Twine &KindDetail);

  // Predicate-chain refusal raised at the IR level. `KernelName` is
  // captured for bucketing; `Detail` names the first failing icmp and
  // constant.
  static llvm::Error crossWavePredicateChain(llvm::StringRef KernelName,
                                             const llvm::Twine &Detail);

  // Safety net for the cross-lane writelane/readlane rewrite path: the
  // classifier flagged the kernel but the rewrite pass rewrote zero
  // sites. The disagreement cannot be resolved without a precise
  // dataflow check, so the lift is refused. Buckets under
  // `CrossWaveLaneIdLeak`.
  static llvm::Error
  crossWaveRewriteOracleDisagreement(llvm::StringRef KernelName,
                                     const llvm::Twine &Detail);

  // `HSA_HOTSWAP_STRICT=1` refusal. `Site` is a short stable label
  // (e.g. `"HWREG_MODE_write"`, `"implicitarg.ptr"`) that callers can
  // bucket on without parsing `Detail`; `Detail` explains why the
  // lowering would silently miscompile.
  static llvm::Error strictUnsafeLowering(llvm::StringRef Mnemonic,
                                          uint64_t Offset, llvm::StringRef Site,
                                          llvm::StringRef Detail);

  // A preloaded hidden kernarg dword has no source-side hidden-arg
  // synthesis. `ByteOffset` locates the preloaded slot; `Detail`
  // explains why.
  static llvm::Error preloadedHiddenArgFailure(llvm::StringRef KernelName,
                                               int ByteOffset,
                                               const llvm::Twine &Detail);

  // A preloaded kernarg byte lands in the source implicit-arg range but
  // has no source hidden-arg metadata mapping; strict mode refuses the
  // target hidden-block fallback.
  static llvm::Error preloadedImplicitArgFailure(llvm::StringRef KernelName,
                                                 int ByteOffset);

  // Kernel descriptor was not parsed from .rodata, so UserSgprLayout
  // cannot be derived. `KernelName` is captured for the diagnostic.
  static llvm::Error missingKernelDescriptor(llvm::StringRef KernelName);

  // Descriptor-derived UserSgprLayout consistency check failed.
  static llvm::Error userSgprLayoutMismatch(llvm::StringRef KernelName,
                                            const llvm::Twine &Detail);

  // Source cluster dimensions are explicit and non-disabled.
  static llvm::Error unsupportedSourceClusterDims(llvm::StringRef KernelName,
                                                  const llvm::Twine &Detail);
};

} // namespace COMGR::hotswap

#endif
