//===- raise_failure.cpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "raise_failure.hpp"

#include "llvm/Support/ErrorHandling.h"

namespace transpiler {

const char *reasonString(RaiseFailureReason R) {
  switch (R) {
  case RaiseFailureReason::None:
    return "None";
  case RaiseFailureReason::BadInput:
    return "BadInput";
  }
  llvm_unreachable("unhandled RaiseFailureReason");
}

} // namespace transpiler
