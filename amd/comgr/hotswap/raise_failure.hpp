//===- raise_failure.hpp - Structured raise-failure values ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP
#define HOTSWAP_TRANSPILER_RAISE_FAILURE_HPP

#include <cstdint>
#include <string>

namespace transpiler {

// Lives in its own header so the handler layer can depend on failure
// values without pulling in the rest of the top-level `raiser.hpp`
// interface.
enum class RaiseFailureReason : uint16_t {
  None = 0,
  BadInput,
};

const char *reasonString(RaiseFailureReason R);

struct RaiseFailure {
  RaiseFailureReason reason = RaiseFailureReason::None;
  // Optional human-readable context.
  std::string detail;

  bool hasFailed() const { return reason != RaiseFailureReason::None; }
};

} // namespace transpiler

#endif
