//===- code_object_utils.hpp - AMDGPU code-object metadata ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_HPP
#define HOTSWAP_TRANSPILER_CODE_OBJECT_UTILS_HPP

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct KernelArgMeta {
  std::string name;
  uint32_t offset = 0;
  uint32_t size = 0;
  std::string valueKind;
  int addressSpace = -1;
};

// Per-kernel metadata extracted from the AMDGPU code object's MsgPack notes
// + kernel descriptor (`<name>.kd`).
struct KernelMeta {
  std::string name;
  uint32_t kernargSegmentSize = 0;
  uint32_t groupSegmentFixedSize = 0;
  uint32_t privateSegmentFixedSize = 0;
  uint32_t maxFlatWorkgroupSize = 256;
  std::vector<KernelArgMeta> args;

  bool hasKernelDescriptor = false;
  uint32_t computePgmRsrc1 = 0;
  uint32_t computePgmRsrc2 = 0;
  uint16_t kernelCodeProperties = 0;
  uint16_t kernargPreload = 0;

  // Byte offset (8-byte aligned) of the first hidden argument in the
  // kernarg segment. Hidden arguments (`hidden_*` value kinds) are
  // appended after every explicit argument.
  uint64_t implicitArgsBase() const {
    uint64_t MaxEnd = 0;
    for (const KernelArgMeta &Arg : args) {
      if (llvm::StringRef(Arg.valueKind).starts_with("hidden_")) {
        continue;
      }
      uint64_t End = static_cast<uint64_t>(Arg.offset) + Arg.size;
      if (End > MaxEnd) {
        MaxEnd = End;
      }
    }
    return llvm::alignTo(MaxEnd, 8);
  }
};

} // namespace transpiler

#endif
