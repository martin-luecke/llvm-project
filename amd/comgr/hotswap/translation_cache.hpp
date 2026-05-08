#ifndef HOTSWAP_TRANSPILER_TRANSLATION_CACHE_HPP
#define HOTSWAP_TRANSPILER_TRANSLATION_CACHE_HPP

#include "pipeline.hpp"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

struct TranslationCacheRequest {
  llvm::ArrayRef<uint8_t> sourceObject;
  std::string sourceGfx;
  std::string targetGfx;
  std::string sourceIsa;
  std::string targetIsa;
  std::string codeIsa;
  std::string hotswapRulesPath;
  std::string cacheDirectory;
  std::string cacheSkipKernels;
  int origMach = -1;
  bool enableWritelaneRewrite = true;
  bool enableWaveNative = true;
  // Diagnostic high-precision MFMA path: when true, bf16 inputs to a
  // WMMA-lowered MFMA are software-upcast to fp32 and routed through a
  // chained `mfma_f32_16x16x4f32` sequence instead of the default
  // `mfma_f32_16x16x16bf16_1k` chain. Default is false (existing
  // behaviour). Threaded through to the raiser via
  // `RaiseContext::enableHighPrecisionMfma` and read by
  // `wmma_lowering.cpp::runGroupPass`. Included in the translation cache
  // key so a flag flip invalidates cached translations.
  bool enableHighPrecisionMfma = false;
  bool strictMode = false;
  bool cacheDisabled = true;
  bool cacheReadonly = false;
};

enum class TranslationCacheStatus {
  Disabled,
  Bypassed,
  Miss,
  Hit,
  Invalid,
  WriteSuccess,
  WriteFailed,
};

struct TranslationCacheLookup {
  TranslationCacheStatus status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string metadataPath;
  std::string objectPath;
  std::string reason;
  PipelineResult result;
};

struct TranslationCacheWrite {
  TranslationCacheStatus status = TranslationCacheStatus::Disabled;
  std::string key;
  std::string metadataPath;
  std::string objectPath;
  std::string reason;
};

const char *translationCacheStatusString(TranslationCacheStatus status);

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &request);

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &request, const PipelineResult &result);

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> kernelNames, llvm::StringRef skipList);

std::string sha256Hex(llvm::ArrayRef<uint8_t> data);

} // namespace transpiler

#endif
