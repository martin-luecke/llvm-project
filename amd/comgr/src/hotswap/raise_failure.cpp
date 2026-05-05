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
  case RaiseFailureReason::None:
    return "None";
  case RaiseFailureReason::BadInput:
    return "BadInput";
  case RaiseFailureReason::UnsupportedOpcode:
    return "UnsupportedOpcode";
  case RaiseFailureReason::UnsupportedShape:
    return "UnsupportedShape";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
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

} // namespace COMGR::hotswap
