//===- error-collector.cpp ------------------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "error-collector.h"

using namespace llvm;

namespace COMGR::hotswap {

void ErrorCollector::addError(Error &&Err) {
  if (Err)
    Errors.push_back(std::move(Err));
}

Error ErrorCollector::makeError() {
  Error Joined = Error::success();
  for (Error &E : Errors)
    Joined = joinErrors(std::move(Joined), std::move(E));
  Errors.clear();
  return Joined;
}

ErrorCollector::~ErrorCollector() {
  for (Error &E : Errors)
    consumeError(std::move(E));
}

} // namespace COMGR::hotswap
