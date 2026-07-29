//===- hotswap-object.h - Source object eligibility -------------*- C++ -*-===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_TOOLS_HOTSWAP_HSA_TOOL_OBJECT_H
#define COMGR_TOOLS_HOTSWAP_HSA_TOOL_OBJECT_H

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>

namespace COMGR::hotswap::hsa_tool {

bool inspectSourceStorage(llvm::ArrayRef<uint8_t> Object,
                          llvm::ArrayRef<std::string> KernelDescriptors,
                          std::string &Failure);

} // namespace COMGR::hotswap::hsa_tool

#endif
