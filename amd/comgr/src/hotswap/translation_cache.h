//===- translation_cache.h - Hotswap transpiler ---------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_TRANSLATION_CACHE_H
#define HOTSWAP_TRANSPILER_TRANSLATION_CACHE_H

#include "pipeline.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace COMGR::hotswap {

struct TranslationCacheRequest {
  llvm::ArrayRef<uint8_t> SourceObject;
  std::string SourceGfx;
  std::string TargetGfx;
  std::string SourceIsa;
  std::string TargetIsa;
  std::string CodeIsa;
  std::string HotswapRulesPath;
  int OrigMach = -1;
  bool EnableWritelaneRewrite = true;
  bool EnableWaveNative = true;
  bool StrictMode = false;
};

enum class TranslationCacheStatus {
  Disabled,
  Miss,
  Hit,
  Invalid,
  WriteSuccess,
  WriteFailed,
};

struct TranslationCacheLookup {
  TranslationCacheStatus Status = TranslationCacheStatus::Disabled;
  std::string Key;
  std::string MetadataPath;
  std::string ObjectPath;
  std::string Reason;
  PipelineResult Result;
};

struct TranslationCacheWrite {
  TranslationCacheStatus Status = TranslationCacheStatus::Disabled;
  std::string Key;
  std::string MetadataPath;
  std::string ObjectPath;
  std::string Reason;
};

const char *translationCacheStatusString(TranslationCacheStatus Status);

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &Request);

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &Request, const PipelineResult &Result);

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> KernelNames);

std::string sha256Hex(llvm::ArrayRef<uint8_t> Data);

} // namespace COMGR::hotswap

#endif
