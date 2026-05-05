//===- comgr-hotswap-transpile.cpp - ISA transpilation via LLVM IR --===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// COMgr entry point for the hotswap transpiler. Where the byte-level
/// `amd_comgr_hotswap_rewrite` path patches a small set of stepping-specific
/// instruction encodings in place, this entry point hands the whole code
/// object to the hotswap pipeline - every kernel is disassembled, raised to
/// LLVM IR, re-lowered through the stock AMDGPU backend for the target ISA,
/// and re-linked into a single merged HSACO via
/// `COMGR::hotswap::runPipelineAllKernels` (see amd/comgr/hotswap/pipeline.hpp and
/// amd/comgr/hotswap/raise_cli.cpp for the standalone driver this entry point
/// mirrors).
///
/// Failure is loud: any per-kernel raise failure surfaced by the hotswap
/// pipeline turns into `AMD_COMGR_STATUS_ERROR`. The hotswap library logs
/// the offending kernel and mnemonic on stderr (use hotswap's CLI with the
/// `--write-hsaco` mode for the same output).
///
//===----------------------------------------------------------------------===//

#include "amd_comgr.h"
#include "comgr.h"

#include "hotswap/code_object_utils.h"
#include "hotswap/pipeline.h"
#include "hotswap/translation_cache.h"

#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <vector>

using namespace COMGR;

namespace {

struct HotswapTranspileResult {
  bool success = false;
  bool cacheHit = false;
  amd_comgr_hotswap_cache_lookup_status_t lookupStatus =
      AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  amd_comgr_hotswap_cache_write_status_t writeStatus =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  int64_t liftedCount = 0;
  int64_t totalCount = 0;
  std::string backend = "comgr";
  std::string sourceGfx;
  std::string targetGfx;
  std::string cacheKey;
  std::string cacheDetail;
  std::string cacheMetadataPath;
  std::string cacheObjectPath;
  std::string failReason;
  std::string failDetail;

  static HotswapTranspileResult *convert(
      amd_comgr_hotswap_transpile_result_t result) {
    return reinterpret_cast<HotswapTranspileResult *>(
        static_cast<uintptr_t>(result.handle));
  }

  static amd_comgr_hotswap_transpile_result_t convert(
      HotswapTranspileResult *result) {
    amd_comgr_hotswap_transpile_result_t handle = {
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result))};
    return handle;
  }
};

amd_comgr_status_t createDataObject(amd_comgr_data_kind_t kind,
                                    llvm::StringRef data,
                                    amd_comgr_data_t *output) {
  DataObject *Object = DataObject::allocate(kind);
  if (!Object)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;

  if (amd_comgr_status_t Status = Object->setData(data)) {
    Object->release();
    return Status;
  }

  *output = DataObject::convert(Object);
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t createExecutableData(llvm::ArrayRef<uint8_t> hsaco,
                                        amd_comgr_data_t *output) {
  return createDataObject(
      AMD_COMGR_DATA_KIND_EXECUTABLE,
      llvm::StringRef(reinterpret_cast<const char *>(hsaco.data()),
                      hsaco.size()),
      output);
}

bool hasFlag(const amd_comgr_hotswap_transpile_options_t *options,
             amd_comgr_hotswap_transpile_option_flags_t flag) {
  return options && (options->flags & static_cast<uint64_t>(flag));
}

std::string pipelineFailReason(const COMGR::hotswap::PipelineResult &pipeline) {
  if (!pipeline.FailReason.empty())
    return pipeline.FailReason;
  if (pipeline.Hsaco.empty())
    return "empty_output";
  return "hotswap_pipeline_failed";
}

std::string pipelineFailDetail(const COMGR::hotswap::PipelineResult &pipeline) {
  if (!pipeline.FailDetail.empty())
    return pipeline.FailDetail;
  if (!pipeline.FailMnemonic.empty())
    return pipeline.FailMnemonic;
  if (!pipeline.FailKernel.empty())
    return pipeline.FailKernel;
  return "hotswap pipeline did not produce a loadable HSACO";
}

amd_comgr_hotswap_cache_lookup_status_t
lookupStatusFromCacheStatus(COMGR::hotswap::TranslationCacheStatus status) {
  switch (status) {
  case COMGR::hotswap::TranslationCacheStatus::Disabled:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  case COMGR::hotswap::TranslationCacheStatus::Bypassed:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_BYPASSED;
  case COMGR::hotswap::TranslationCacheStatus::Miss:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_MISS;
  case COMGR::hotswap::TranslationCacheStatus::Hit:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_HIT;
  case COMGR::hotswap::TranslationCacheStatus::Invalid:
    return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID;
  case COMGR::hotswap::TranslationCacheStatus::WriteSuccess:
  case COMGR::hotswap::TranslationCacheStatus::WriteFailed:
    break;
  }
  return AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID;
}

amd_comgr_hotswap_cache_write_status_t
writeStatusFromCacheStatus(COMGR::hotswap::TranslationCacheStatus status) {
  switch (status) {
  case COMGR::hotswap::TranslationCacheStatus::WriteSuccess:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_SUCCESS;
  case COMGR::hotswap::TranslationCacheStatus::WriteFailed:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_FAILED;
  case COMGR::hotswap::TranslationCacheStatus::Disabled:
  case COMGR::hotswap::TranslationCacheStatus::Bypassed:
  case COMGR::hotswap::TranslationCacheStatus::Miss:
  case COMGR::hotswap::TranslationCacheStatus::Hit:
  case COMGR::hotswap::TranslationCacheStatus::Invalid:
    return AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  }
  return AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
}

void fillResult(HotswapTranspileResult &result, llvm::StringRef sourceGfx,
                llvm::StringRef targetGfx, bool success, bool cacheHit,
                amd_comgr_hotswap_cache_lookup_status_t lookupStatus,
                amd_comgr_hotswap_cache_write_status_t writeStatus,
                llvm::StringRef cacheDetail,
                const COMGR::hotswap::PipelineResult *pipeline,
                llvm::StringRef cacheKey = "",
                llvm::StringRef cacheMetadataPath = "",
                llvm::StringRef cacheObjectPath = "",
                llvm::StringRef failReason = "",
                llvm::StringRef failDetail = "") {
  result.sourceGfx = sourceGfx.str();
  result.targetGfx = targetGfx.str();
  result.success = success;
  result.cacheHit = cacheHit;
  result.lookupStatus = lookupStatus;
  result.writeStatus = writeStatus;
  result.cacheDetail = cacheDetail.str();
  result.cacheKey = cacheKey.str();
  result.cacheMetadataPath = cacheMetadataPath.str();
  result.cacheObjectPath = cacheObjectPath.str();
  result.failReason = failReason.str();
  result.failDetail = failDetail.str();
  if (pipeline) {
    result.liftedCount = pipeline->LiftedCount;
    result.totalCount = pipeline->TotalCount;
  }
}

amd_comgr_status_t returnResult(HotswapTranspileResult &&value,
                                amd_comgr_hotswap_transpile_result_t *result) {
  if (!result)
    return AMD_COMGR_STATUS_SUCCESS;
  HotswapTranspileResult *owned =
      new (std::nothrow) HotswapTranspileResult(std::move(value));
  if (!owned)
    return AMD_COMGR_STATUS_ERROR_OUT_OF_RESOURCES;
  *result = HotswapTranspileResult::convert(owned);
  return AMD_COMGR_STATUS_SUCCESS;
}

} // namespace

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_with_options(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name,
    const amd_comgr_hotswap_transpile_options_t *options,
    amd_comgr_data_t *output,
    amd_comgr_hotswap_transpile_result_t *result) {
  DataObject *InputP = DataObject::convert(input);
  if (!InputP || !InputP->Data ||
      InputP->DataKind != AMD_COMGR_DATA_KIND_EXECUTABLE || !source_isa_name ||
      !target_isa_name || !output)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  if (options && options->size < sizeof(amd_comgr_hotswap_transpile_options_t))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Validate both ISA names through the same parser the byte-level
  // `amd_comgr_hotswap_rewrite` uses, so the public contract is identical:
  // malformed identifiers are rejected up-front and never reach the hotswap
  // pipeline. We do not gate on the processor name here — hotswap decides
  // per-kernel whether the source/target pair is supported, and surfaces
  // unsupported instructions as a pipeline failure (see
  // RaiseFailure::reason in amd/comgr/hotswap/raise_failure.hpp).
  TargetIdentifier SourceIdent, TargetIdent;
  if (parseTargetIdentifier(source_isa_name, SourceIdent) ||
      parseTargetIdentifier(target_isa_name, TargetIdent))
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  // Hotswap takes the code object by `std::vector<uint8_t>` (see
  // pipeline.hpp). DataObject stores its bytes in a `char *` buffer; copy
  // into the hotswap-shaped container rather than reinterpret-casting the
  // pointer, since the pipeline reads through this buffer many times across
  // kernels and the temporary lifetime needs to be unambiguous.
  const auto *InputBegin = reinterpret_cast<const uint8_t *>(InputP->Data);
  std::vector<uint8_t> InputBytes(InputBegin, InputBegin + InputP->Size);

  COMGR::hotswap::TranslationCacheRequest CacheRequest;
  CacheRequest.SourceObject = llvm::ArrayRef<uint8_t>(InputBytes);
  CacheRequest.SourceGfx = SourceIdent.Processor.str();
  CacheRequest.TargetGfx = TargetIdent.Processor.str();
  CacheRequest.SourceIsa = source_isa_name;
  CacheRequest.TargetIsa = target_isa_name;
  CacheRequest.CodeIsa = source_isa_name;
  CacheRequest.HotswapRulesPath =
      options && options->hotswap_rules_path ? options->hotswap_rules_path : "";
  CacheRequest.CacheDirectory =
      options && options->cache_directory ? options->cache_directory : "";
  CacheRequest.CacheSkipKernels =
      options && options->cache_skip_kernels ? options->cache_skip_kernels : "";
  CacheRequest.StrictMode =
      hasFlag(options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_STRICT);
  CacheRequest.CacheDisabled =
      !options || hasFlag(options,
                          AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_DISABLE) ||
      CacheRequest.CacheDirectory.empty();
  CacheRequest.CacheReadonly =
      hasFlag(options, AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_CACHE_READONLY);

  const std::vector<std::string> KernelNames =
      COMGR::hotswap::listKernelNames(InputBytes);
  const std::string SkippedKernel =
      COMGR::hotswap::skippedKernelForTranslationCache(
          KernelNames, CacheRequest.CacheSkipKernels);

  COMGR::hotswap::TranslationCacheStatus CacheStatus =
      COMGR::hotswap::TranslationCacheStatus::Disabled;
  std::string CacheDetail;
  std::string CacheKey;
  std::string CacheMetadataPath;
  std::string CacheObjectPath;
  bool CacheHit = false;

  COMGR::hotswap::PipelineResult Pipeline;
  if (!SkippedKernel.empty()) {
    CacheStatus = COMGR::hotswap::TranslationCacheStatus::Bypassed;
    CacheDetail = "kernel listed in HSA_HOTSWAP_CACHE_SKIP_KERNELS: " +
                  SkippedKernel;
  } else {
    COMGR::hotswap::TranslationCacheLookup Lookup =
        COMGR::hotswap::lookupTranslationCache(CacheRequest);
    CacheStatus = Lookup.Status;
    CacheDetail = Lookup.Reason;
    CacheKey = Lookup.Key;
    CacheMetadataPath = Lookup.MetadataPath;
    CacheObjectPath = Lookup.ObjectPath;

    if (Lookup.Status == COMGR::hotswap::TranslationCacheStatus::Invalid) {
      HotswapTranspileResult Result;
      fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, false,
                 false, lookupStatusFromCacheStatus(Lookup.Status),
                 AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED, Lookup.Reason,
                 nullptr, Lookup.Key, Lookup.MetadataPath, Lookup.ObjectPath,
                 "cache_invalid", Lookup.Reason);
      if (amd_comgr_status_t ResultStatus =
              returnResult(std::move(Result), result))
        return ResultStatus;
      return AMD_COMGR_STATUS_ERROR;
    }

    if (Lookup.Status == COMGR::hotswap::TranslationCacheStatus::Hit) {
      Pipeline = std::move(Lookup.Result);
      CacheHit = true;
    }
  }

  // Drive the same all-kernels merge path that raise_cli.cpp's --write-hsaco
  // mode falls back on for whole-file flows. We pass hotswap's defaults for
  // the writelane / wave-native toggles (both on, post-graduation) — the
  // public comgr surface intentionally hides those knobs since they are
  // either correctness-preserving rewrites (writelane) or projection
  // strategies (wave-native) that callers should not have to reason about.
  // If an opt-out is ever needed at the comgr boundary it should land as a
  // separate options struct rather than overloading this entry point.
  if (!CacheHit) {
    COMGR::hotswap::ScopedStrictMode StrictMode(CacheRequest.StrictMode);
    Pipeline = COMGR::hotswap::runPipelineAllKernels(InputBytes,
                                                 SourceIdent.Processor.str(),
                                                 TargetIdent.Processor.str());
  }

  if (!Pipeline.Success || Pipeline.Hsaco.empty()) {
    HotswapTranspileResult Result;
    fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, false,
               CacheHit, lookupStatusFromCacheStatus(CacheStatus),
               AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED, CacheDetail,
               &Pipeline, CacheKey, CacheMetadataPath, CacheObjectPath,
               pipelineFailReason(Pipeline), pipelineFailDetail(Pipeline));
    if (amd_comgr_status_t ResultStatus =
            returnResult(std::move(Result), result))
      return ResultStatus;
    return AMD_COMGR_STATUS_ERROR;
  }

  amd_comgr_hotswap_cache_write_status_t CacheWriteStatus =
      AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  if (!CacheHit && CacheStatus == COMGR::hotswap::TranslationCacheStatus::Miss) {
    COMGR::hotswap::TranslationCacheWrite Write =
        COMGR::hotswap::writeTranslationCache(CacheRequest, Pipeline);
    CacheWriteStatus = writeStatusFromCacheStatus(Write.Status);
    if (!Write.Key.empty())
      CacheKey = Write.Key;
    if (!Write.MetadataPath.empty())
      CacheMetadataPath = Write.MetadataPath;
    if (!Write.ObjectPath.empty())
      CacheObjectPath = Write.ObjectPath;
    if (!Write.Reason.empty())
      CacheDetail = Write.Reason;
    if (Write.Status == COMGR::hotswap::TranslationCacheStatus::WriteFailed) {
      HotswapTranspileResult Result;
      fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, false,
                 false, lookupStatusFromCacheStatus(CacheStatus),
                 CacheWriteStatus, Write.Reason, &Pipeline, Write.Key,
                 Write.MetadataPath, Write.ObjectPath, "cache_write_failed",
                 Write.Reason);
      if (amd_comgr_status_t ResultStatus =
              returnResult(std::move(Result), result))
        return ResultStatus;
      return AMD_COMGR_STATUS_ERROR;
    }
  }

  amd_comgr_data_t OutputData = {0};
  if (auto Status = createExecutableData(Pipeline.Hsaco, &OutputData))
    return Status;

  HotswapTranspileResult Result;
  fillResult(Result, CacheRequest.SourceGfx, CacheRequest.TargetGfx, true,
             CacheHit, lookupStatusFromCacheStatus(CacheStatus),
             CacheWriteStatus, CacheDetail, &Pipeline, CacheKey,
             CacheMetadataPath, CacheObjectPath);
  if (amd_comgr_status_t ResultStatus =
          returnResult(std::move(Result), result)) {
    amd_comgr_release_data(OutputData);
    return ResultStatus;
  }

  *output = OutputData;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile(
    amd_comgr_data_t input, const char *source_isa_name,
    const char *target_isa_name, amd_comgr_data_t *output) {
  return amd_comgr_hotswap_transpile_with_options(
      input, source_isa_name, target_isa_name, nullptr, output, nullptr);
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_destroy_hotswap_transpile_result(
    amd_comgr_hotswap_transpile_result_t result) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  delete Result;
  return AMD_COMGR_STATUS_SUCCESS;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_result_get_info(
    amd_comgr_hotswap_transpile_result_t result,
    amd_comgr_hotswap_transpile_result_info_t info, void *value) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result || !value)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  switch (info) {
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SUCCESS:
    *static_cast<bool *>(value) = Result->success;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT:
    *static_cast<bool *>(value) = Result->cacheHit;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_LOOKUP:
    *static_cast<amd_comgr_hotswap_cache_lookup_status_t *>(value) =
        Result->lookupStatus;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_WRITE:
    *static_cast<amd_comgr_hotswap_cache_write_status_t *>(value) =
        Result->writeStatus;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_LIFTED_COUNT:
    *static_cast<int64_t *>(value) = Result->liftedCount;
    return AMD_COMGR_STATUS_SUCCESS;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TOTAL_COUNT:
    *static_cast<int64_t *>(value) = Result->totalCount;
    return AMD_COMGR_STATUS_SUCCESS;
  }
  return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
}

amd_comgr_status_t AMD_COMGR_API amd_comgr_hotswap_transpile_result_get_string(
    amd_comgr_hotswap_transpile_result_t result,
    amd_comgr_hotswap_transpile_result_string_t field, size_t *size,
    char *value) {
  HotswapTranspileResult *Result = HotswapTranspileResult::convert(result);
  if (!Result || !size)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  const std::string *Field = nullptr;
  switch (field) {
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_BACKEND:
    Field = &Result->backend;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SOURCE_GFX:
    Field = &Result->sourceGfx;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TARGET_GFX:
    Field = &Result->targetGfx;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_KEY:
    Field = &Result->cacheKey;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_DETAIL:
    Field = &Result->cacheDetail;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_METADATA_PATH:
    Field = &Result->cacheMetadataPath;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_OBJECT_PATH:
    Field = &Result->cacheObjectPath;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_REASON:
    Field = &Result->failReason;
    break;
  case AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL:
    Field = &Result->failDetail;
    break;
  }
  if (!Field)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;

  const size_t Required = Field->size() + 1;
  if (!value) {
    *size = Required;
    return AMD_COMGR_STATUS_SUCCESS;
  }
  if (*size < Required)
    return AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT;
  std::memcpy(value, Field->c_str(), Required);
  *size = Required;
  return AMD_COMGR_STATUS_SUCCESS;
}
