//===- raise_failure.cpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise_failure.h"

#include "decoded_inst.h"

namespace COMGR::hotswap {

const char *reasonString(RaiseFailureReason R) {
  switch (R) {
  case RaiseFailureReason::None:                    return "None";
  case RaiseFailureReason::BadInput:                return "BadInput";
  case RaiseFailureReason::UnsupportedOpcode:       return "UnsupportedOpcode";
  case RaiseFailureReason::UnsupportedShape:        return "UnsupportedShape";
  case RaiseFailureReason::CrossWaveLaneIdLeak:
    return "cross-wave-lane-id-leak";
  case RaiseFailureReason::CrossWaveUnrewritableShuffle:
    return "cross-wave-unrewritable-shuffle";
  case RaiseFailureReason::CrossWaveShuffleRewritePending:
    return "cross-wave-shuffle-rewrite-pending";
  case RaiseFailureReason::CrossWaveReplicaRace:
    return "cross-wave-replica-race";
  case RaiseFailureReason::CrossWaveLanePredicatedExec:
    return "cross-wave-lane-predicated-exec";
  case RaiseFailureReason::CrossWavePredicateChain:
    return "cross-wave-predicate-chain";
  case RaiseFailureReason::StrictUnsafeLowering:
    return "strict-unsafe-lowering";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
}

RaiseFailure RaiseFailure::unsupportedShape(const DecodedInst &Di,
                                             llvm::StringRef Format,
                                             const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedShape;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

RaiseFailure RaiseFailure::unsupportedOpcode(const DecodedInst &Di,
                                              llvm::StringRef Format) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::UnsupportedOpcode;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Format.str();
  F.Offset = Di.Offset;
  return F;
}

// ----------------------------------------------------------------------------
// Phase 1.4.5 wave-size-obstruction factories. All share the same
// structure: take the refused instruction for mnemonic / offset, and
// a kind-specific detail string for the `detail` field.
// ----------------------------------------------------------------------------

namespace {

RaiseFailure makeCrossWaveFailure(RaiseFailureReason Reason,
                                   const DecodedInst &Di,
                                   const llvm::Twine &KindDetail) {
  RaiseFailure F;
  F.Reason = Reason;
  F.Mnemonic = Di.Mnemonic;
  F.Format = reasonString(Reason);
  F.Offset = Di.Offset;
  F.Detail = KindDetail.str();
  return F;
}

} // namespace

RaiseFailure RaiseFailure::crossWaveLaneIdLeak(const DecodedInst &Di,
                                                const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveLaneIdLeak, Di,
                               KindDetail);
}

RaiseFailure RaiseFailure::crossWaveUnrewritableShuffle(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveUnrewritableShuffle, Di, KindDetail);
}

RaiseFailure RaiseFailure::crossWaveShuffleRewritePending(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveShuffleRewritePending, Di, KindDetail);
}

RaiseFailure RaiseFailure::crossWaveReplicaRace(const DecodedInst &Di,
                                                 const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(RaiseFailureReason::CrossWaveReplicaRace, Di,
                               KindDetail);
}

RaiseFailure RaiseFailure::crossWaveLanePredicatedExec(
    const DecodedInst &Di, const llvm::Twine &KindDetail) {
  return makeCrossWaveFailure(
      RaiseFailureReason::CrossWaveLanePredicatedExec, Di, KindDetail);
}

// see hotswap/docs/modrep-predicate-chain.md §5 (narrow-O1)
RaiseFailure RaiseFailure::crossWavePredicateChain(
    llvm::StringRef KernelName, const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::CrossWavePredicateChain;
  F.Mnemonic = "workitem.id.x-predicate-chain-classifier";
  F.Format = reasonString(RaiseFailureReason::CrossWavePredicateChain);
  F.Offset = 0;
  F.Detail = ("kernel '" + KernelName + "': " + Detail).str();
  return F;
}

RaiseFailure RaiseFailure::strictUnsafeLowering(const DecodedInst &Di,
                                                  llvm::StringRef Site,
                                                  const llvm::Twine &Detail) {
  RaiseFailure F;
  F.Reason = RaiseFailureReason::StrictUnsafeLowering;
  F.Mnemonic = Di.Mnemonic;
  F.Format = Site.str();
  F.Offset = Di.Offset;
  F.Detail = Detail.str();
  return F;
}

} // namespace COMGR::hotswap
