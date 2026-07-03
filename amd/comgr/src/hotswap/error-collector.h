//===- error-collector.h ---------------------------------------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Accumulates llvm::Errors so several failing operations can be reported in
/// aggregate. Any errors still held at destruction are consumed, so callers
/// need not drain the collector on every path.
///
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_ERROR_COLLECTOR_H
#define HOTSWAP_TRANSPILER_ERROR_COLLECTOR_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

class ErrorCollector {
public:
  ErrorCollector() = default;
  ErrorCollector(const ErrorCollector &) = delete;
  ErrorCollector &operator=(const ErrorCollector &) = delete;
  ~ErrorCollector();

  /// Stores E if it represents a failure; a success value is discarded.
  void addError(llvm::Error &&E);

  /// Returns a single error joining all stored errors and empties the
  /// collector. Returns success when nothing was collected.
  llvm::Error makeError();

private:
  llvm::SmallVector<llvm::Error> Errors;
};

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_ERROR_COLLECTOR_H
