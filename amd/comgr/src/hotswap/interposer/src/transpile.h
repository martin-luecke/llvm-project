//===- transpile.h - COMGR hotswap transpile wrapper ---------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Thin wrapper over the COMGR hotswap transpiler C API.
///
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_INTERPOSER_TRANSPILE_H_
#define HOTSWAP_INTERPOSER_TRANSPILE_H_

#include <cstddef>

namespace hotswap {
namespace interposer {

/// Transpile an AMDGPU code-object ELF from \p SourceIsa to \p TargetIsa via
/// COMGR (e.g. "amdgcn-amd-amdhsa--gfx1250" -> "amdgcn-amd-amdhsa--gfx1151").
/// On success returns 0 and sets \p OutData (malloc'd, caller frees) and
/// \p OutSize. Returns non-zero on failure with \p OutData left null.
int retargetCodeObject(const void *ElfData, size_t ElfSize,
                       const char *SourceIsa, const char *TargetIsa,
                       void **OutData, size_t *OutSize);

} // namespace interposer
} // namespace hotswap

#endif // HOTSWAP_INTERPOSER_TRANSPILE_H_
