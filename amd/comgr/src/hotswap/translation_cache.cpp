//===- translation_cache.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "translation_cache.h"

#include "code_object_utils.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>

#define DEBUG_TYPE "translation-cache"

#ifndef LLVM_TOOLS_DIR
#define LLVM_TOOLS_DIR "/usr/bin"
#endif

namespace COMGR::hotswap {
namespace {

constexpr int kCacheSchemaVersion = 1;

struct FileIdentity {
  std::string path;
  bool present = false;
  uint64_t size = 0;
  int64_t mtimeSec = 0;
  int64_t mtimeNsec = 0;
  std::string sha256;
  std::string error;
};

struct KeyData {
  std::string Key;
  std::string SourceSha256;
  std::string RulesSha256;
  std::string RulesError;
  std::string BuildIdentity;
  std::string LlcIdentity;
  std::string LlvmMcIdentity;
  std::string LldIdentity;
  std::string ElfMachineHex;
  std::string ElfFlagsHex;
  std::vector<std::string> KernelNames;
  std::string Error;
};

std::string hexU32(uint32_t value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "0x" << llvm::format_hex_no_prefix(value, 0);
  return os.str();
}

bool readElfHeaderFields(llvm::ArrayRef<uint8_t> data, uint16_t &machine,
                         uint32_t &flags) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(data.data()),
                      data.size()),
      "", false);
  auto objOrErr = llvm::object::ObjectFile::createELFObjectFile(*buf);
  if (!objOrErr) {
    (void)llvm::toString(objOrErr.takeError());
    return false;
  }
  const auto *elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(objOrErr->get());
  if (!elf)
    return false;
  machine = elf->getEMachine();
  flags = elf->getPlatformFlags();
  return true;
}

std::string hashFile(llvm::StringRef path, std::string &error) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = buffer.getError().message();
    return "";
  }
  llvm::StringRef contents = (*buffer)->getBuffer();
  return sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(contents.data()), contents.size()));
}

FileIdentity statIdentity(llvm::StringRef path) {
  FileIdentity id;
  id.path = path.str();
  llvm::sys::fs::file_status st;
  if (llvm::sys::fs::status(path, st))
    return id;
  if (!llvm::sys::fs::exists(st))
    return id;
  id.present = true;
  id.size = static_cast<uint64_t>(st.getSize());
  // TimePoint<> is std::chrono::time_point<system_clock, nanoseconds> on every
  // platform, so split into seconds + nanoseconds-of-second once.
  auto sinceEpoch = st.getLastModificationTime().time_since_epoch();
  auto secs = std::chrono::duration_cast<std::chrono::seconds>(sinceEpoch);
  id.mtimeSec = secs.count();
  id.mtimeNsec =
      std::chrono::duration_cast<std::chrono::nanoseconds>(sinceEpoch - secs)
          .count();
  id.sha256 = hashFile(id.path, id.error);
  return id;
}

std::string identityString(const FileIdentity &id) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << id.path << "|present=" << (id.present ? "1" : "0")
     << "|size=" << id.size << "|mtime=" << id.mtimeSec << "."
     << id.mtimeNsec << "|sha256=" << id.sha256;
  if (!id.error.empty())
    os << "|error=" << id.error;
  return os.str();
}

std::string loadedImageIdentity() {
  static const std::string identity = [] {
    std::string out;
    llvm::raw_string_ostream os(out);
    os << "llvm=" << LLVM_VERSION_STRING;
    // getMainExecutable uses dladdr/Mach-O lookup on Unix and
    // GetModuleFileName on Windows; either way we get the filesystem path
    // of the loaded image so its mtime/sha256 can key the cache.
    std::string image = llvm::sys::fs::getMainExecutable(
        nullptr, reinterpret_cast<void *>(&loadedImageIdentity));
    if (!image.empty())
      os << "|image=" << identityString(statIdentity(image));
    else
      os << "|image=<unavailable>";
    return os.str();
  }();
  return identity;
}

void appendKeyField(std::string &material, llvm::StringRef name,
                    llvm::StringRef value) {
  material.append(name.data(), name.size());
  material.push_back('\0');
  material += std::to_string(value.size());
  material.push_back(':');
  if (!value.empty())
    material.append(value.data(), value.size());
  material.push_back('\0');
}

void appendKeyField(std::string &material, llvm::StringRef name, bool value) {
  appendKeyField(material, name, llvm::StringRef(value ? "true" : "false"));
}

void appendKeyField(std::string &material, llvm::StringRef name, int value) {
  appendKeyField(material, name, std::to_string(value));
}

const FileIdentity &llcIdentity() {
  static const FileIdentity identity =
      statIdentity(std::string(LLVM_TOOLS_DIR) + "/llc");
  return identity;
}

const FileIdentity &llvmMcIdentity() {
  static const FileIdentity identity =
      statIdentity(std::string(LLVM_TOOLS_DIR) + "/llvm-mc");
  return identity;
}

const FileIdentity &lldIdentity() {
  static const FileIdentity identity =
      statIdentity(std::string(LLVM_TOOLS_DIR) + "/ld.lld");
  return identity;
}

KeyData buildKeyData(const TranslationCacheRequest &request) {
  KeyData data;
  if (request.SourceObject.empty()) {
    data.Error = "empty source code object";
    return data;
  }
  if (request.SourceGfx.empty() || request.TargetGfx.empty()) {
    data.Error = "missing source or target gfx";
    return data;
  }

  data.SourceSha256 = sha256Hex(request.SourceObject);
  uint16_t machine = 0;
  uint32_t flags = 0;
  if (readElfHeaderFields(request.SourceObject, machine, flags)) {
    data.ElfMachineHex = hexU32(machine);
    data.ElfFlagsHex = hexU32(flags);
  } else {
    data.Error = "source code object is not a 64-bit little-endian ELF";
    return data;
  }

  if (!request.HotswapRulesPath.empty()) {
    data.RulesSha256 = hashFile(request.HotswapRulesPath, data.RulesError);
    if (!data.RulesError.empty()) {
      data.Error = "failed to hash HSA_HOTSWAP_RULES '" +
                   request.HotswapRulesPath + "': " + data.RulesError;
      return data;
    }
  }

  const std::string toolsDir = LLVM_TOOLS_DIR;
  const FileIdentity &llc = llcIdentity();
  const FileIdentity &llvmMc = llvmMcIdentity();
  const FileIdentity &lld = lldIdentity();
  if (!llc.present || !llvmMc.present || !lld.present || !llc.error.empty() ||
      !llvmMc.error.empty() || !lld.error.empty()) {
    data.Error = "LLVM tool identity is incomplete under " + toolsDir;
    return data;
  }
  data.LlcIdentity = identityString(llc);
  data.LlvmMcIdentity = identityString(llvmMc);
  data.LldIdentity = identityString(lld);
  data.BuildIdentity = loadedImageIdentity();
  data.KernelNames = listKernelNames(
      std::vector<uint8_t>(request.SourceObject.begin(),
                           request.SourceObject.end()));

  std::string material;
  appendKeyField(material, "schema", std::to_string(kCacheSchemaVersion));
  appendKeyField(material, "source_sha256", data.SourceSha256);
  appendKeyField(material, "source_gfx", request.SourceGfx);
  appendKeyField(material, "target_gfx", request.TargetGfx);
  appendKeyField(material, "source_isa", request.SourceIsa);
  appendKeyField(material, "target_isa", request.TargetIsa);
  appendKeyField(material, "code_isa", request.CodeIsa);
  appendKeyField(material, "elf_machine", data.ElfMachineHex);
  appendKeyField(material, "elf_flags", data.ElfFlagsHex);
  appendKeyField(material, "orig_mach", request.OrigMach);
  appendKeyField(material, "rules_path", request.HotswapRulesPath);
  appendKeyField(material, "rules_sha256", data.RulesSha256);
  appendKeyField(material, "strict", request.StrictMode);
  appendKeyField(material, "enable_writelane_rewrite",
                 request.EnableWritelaneRewrite);
  appendKeyField(material, "enable_wave_native", request.EnableWaveNative);
  appendKeyField(material, "hotswap_build_identity", data.BuildIdentity);
  appendKeyField(material, "llc_identity", data.LlcIdentity);
  appendKeyField(material, "llvm_mc_identity", data.LlvmMcIdentity);
  appendKeyField(material, "lld_identity", data.LldIdentity);
  data.Key = sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(material.data()), material.size()));
  return data;
}

std::string cacheRoot(const TranslationCacheRequest &request) {
  return request.CacheDirectory;
}

bool cacheDisabledByPolicy(const TranslationCacheRequest &request) {
  return request.CacheDisabled || cacheRoot(request).empty();
}

std::string cacheSubdir(const TranslationCacheRequest &request,
                        llvm::StringRef key) {
  llvm::SmallString<256> path(cacheRoot(request));
  llvm::sys::path::append(path, key.substr(0, 2));
  return std::string(path);
}

std::string cacheObjectPath(const TranslationCacheRequest &request,
                            llvm::StringRef key) {
  llvm::SmallString<256> path(cacheSubdir(request, key));
  llvm::sys::path::append(path, llvm::Twine(key) + ".hsaco");
  return std::string(path);
}

std::string cacheMetadataPath(const TranslationCacheRequest &request,
                              llvm::StringRef key) {
  llvm::SmallString<256> path(cacheSubdir(request, key));
  llvm::sys::path::append(path, llvm::Twine(key) + ".json");
  return std::string(path);
}

bool exists(llvm::StringRef path) {
  return llvm::sys::fs::exists(path);
}

std::string jsonToString(llvm::json::Value value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  value.print(os);
  os << "\n";
  return os.str();
}

bool writeFileAtomic(llvm::StringRef path, llvm::StringRef contents,
                     std::string &error) {
  llvm::SmallString<256> model(path);
  model += ".tmp-%%%%%%";
  llvm::SmallString<256> tmpPath;
  int fd = -1;
  if (auto ec = llvm::sys::fs::createUniqueFile(model, fd, tmpPath)) {
    error = ec.message();
    return false;
  }
  {
    llvm::raw_fd_ostream os(fd, true);
    os << contents;
    if (os.has_error()) {
      error = os.error().message();
      os.clear_error();
      llvm::sys::fs::remove(tmpPath);
      return false;
    }
  }
  if (auto ec = llvm::sys::fs::rename(tmpPath, path)) {
    error = ec.message();
    llvm::sys::fs::remove(tmpPath);
    return false;
  }
  return true;
}

bool writeFileAtomic(llvm::StringRef path, llvm::ArrayRef<uint8_t> data,
                     std::string &error) {
  return writeFileAtomic(
      path, llvm::StringRef(reinterpret_cast<const char *>(data.data()),
                            data.size()),
      error);
}

std::optional<std::string> requireString(const llvm::json::Object &obj,
                                         llvm::StringRef field,
                                         std::string &reason) {
  auto value = obj.getString(field);
  if (!value) {
    reason = "metadata field '" + field.str() + "' missing or not a string";
    return std::nullopt;
  }
  return value->str();
}

std::optional<int64_t> requireInt(const llvm::json::Object &obj,
                                  llvm::StringRef field, std::string &reason) {
  auto value = obj.getInteger(field);
  if (!value) {
    reason = "metadata field '" + field.str() + "' missing or not an integer";
    return std::nullopt;
  }
  return *value;
}

std::optional<bool> requireBool(const llvm::json::Object &obj,
                                llvm::StringRef field, std::string &reason) {
  auto value = obj.getBoolean(field);
  if (!value) {
    reason = "metadata field '" + field.str() + "' missing or not a boolean";
    return std::nullopt;
  }
  return *value;
}

bool requireEqualString(const llvm::json::Object &obj, llvm::StringRef field,
                        llvm::StringRef expected, std::string &reason) {
  auto value = requireString(obj, field, reason);
  if (!value)
    return false;
  if (*value != expected) {
    reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualInt(const llvm::json::Object &obj, llvm::StringRef field,
                     int64_t expected, std::string &reason) {
  auto value = requireInt(obj, field, reason);
  if (!value)
    return false;
  if (*value != expected) {
    reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualBool(const llvm::json::Object &obj, llvm::StringRef field,
                      bool expected, std::string &reason) {
  auto value = requireBool(obj, field, reason);
  if (!value)
    return false;
  if (*value != expected) {
    reason = "metadata field '" + field.str() + "' mismatch";
    return false;
  }
  return true;
}

llvm::json::Array kernelArray(const std::vector<std::string> &kernelNames) {
  llvm::json::Array arr;
  for (llvm::StringRef name : kernelNames)
    arr.push_back(name);
  return arr;
}

bool validateKernelArray(const llvm::json::Object &obj,
                         const std::vector<std::string> &expected,
                         std::string &reason) {
  const llvm::json::Array *arr = obj.getArray("kernel_names");
  if (!arr) {
    reason = "metadata field 'kernel_names' missing or not an array";
    return false;
  }
  if (arr->size() != expected.size()) {
    reason = "metadata kernel_names size mismatch";
    return false;
  }
  for (size_t i = 0; i < expected.size(); ++i) {
    auto value = (*arr)[i].getAsString();
    if (!value || *value != expected[i]) {
      reason = "metadata kernel_names mismatch";
      return false;
    }
  }
  return true;
}

llvm::json::Object metadataObject(const TranslationCacheRequest &request,
                                  const KeyData &keyData,
                                  const PipelineResult &result,
                                  llvm::StringRef objectSha256) {
  return llvm::json::Object{
      {"schema_version", kCacheSchemaVersion},
      {"Key", keyData.Key},
      {"source_object_sha256", keyData.SourceSha256},
      {"source_gfx", request.SourceGfx},
      {"target_gfx", request.TargetGfx},
      {"source_isa", request.SourceIsa},
      {"target_isa", request.TargetIsa},
      {"code_isa", request.CodeIsa},
      {"elf_machine", keyData.ElfMachineHex},
      {"elf_flags", keyData.ElfFlagsHex},
      {"orig_mach", request.OrigMach},
      {"hotswap_rules_path", request.HotswapRulesPath},
      {"hotswap_rules_sha256", keyData.RulesSha256},
      {"strict_mode", request.StrictMode},
      {"enable_writelane_rewrite", request.EnableWritelaneRewrite},
      {"enable_wave_native", request.EnableWaveNative},
      {"hotswap_build_identity", keyData.BuildIdentity},
      {"llc_identity", keyData.LlcIdentity},
      {"llvm_mc_identity", keyData.LlvmMcIdentity},
      {"lld_identity", keyData.LldIdentity},
      {"kernel_count", static_cast<int64_t>(keyData.KernelNames.size())},
      {"kernel_names", kernelArray(keyData.KernelNames)},
      {"cached_object_sha256", objectSha256.str()},
      {"cached_object_size", static_cast<int64_t>(result.Hsaco.size())},
      {"lifted_count", result.LiftedCount},
      {"total_count", result.TotalCount},
      {"uses_scratch_private_segment", result.UsesScratchPrivateSegment},
      {"source_private_segment_fixed_size",
       static_cast<int64_t>(result.SourcePrivateSegmentFixedSize)},
      {"target_private_segment_fixed_size",
       static_cast<int64_t>(result.TargetPrivateSegmentFixedSize)},
      {"target_enable_private_segment", result.TargetEnablePrivateSegment},
  };
}

bool validateMetadata(const TranslationCacheRequest &request,
                      const KeyData &keyData, const llvm::json::Object &obj,
                      llvm::StringRef objectSha256, size_t objectSize,
                      PipelineResult &result, std::string &reason) {
  if (!requireEqualInt(obj, "schema_version", kCacheSchemaVersion, reason) ||
      !requireEqualString(obj, "Key", keyData.Key, reason) ||
      !requireEqualString(obj, "source_object_sha256", keyData.SourceSha256,
                          reason) ||
      !requireEqualString(obj, "source_gfx", request.SourceGfx, reason) ||
      !requireEqualString(obj, "target_gfx", request.TargetGfx, reason) ||
      !requireEqualString(obj, "source_isa", request.SourceIsa, reason) ||
      !requireEqualString(obj, "target_isa", request.TargetIsa, reason) ||
      !requireEqualString(obj, "code_isa", request.CodeIsa, reason) ||
      !requireEqualString(obj, "elf_machine", keyData.ElfMachineHex, reason) ||
      !requireEqualString(obj, "elf_flags", keyData.ElfFlagsHex, reason) ||
      !requireEqualInt(obj, "orig_mach", request.OrigMach, reason) ||
      !requireEqualString(obj, "hotswap_rules_path", request.HotswapRulesPath,
                          reason) ||
      !requireEqualString(obj, "hotswap_rules_sha256", keyData.RulesSha256,
                          reason) ||
      !requireEqualBool(obj, "strict_mode", request.StrictMode, reason) ||
      !requireEqualBool(obj, "enable_writelane_rewrite",
                        request.EnableWritelaneRewrite, reason) ||
      !requireEqualBool(obj, "enable_wave_native", request.EnableWaveNative,
                        reason) ||
      !requireEqualString(obj, "hotswap_build_identity",
                          keyData.BuildIdentity, reason) ||
      !requireEqualString(obj, "llc_identity", keyData.LlcIdentity, reason) ||
      !requireEqualString(obj, "llvm_mc_identity", keyData.LlvmMcIdentity,
                          reason) ||
      !requireEqualString(obj, "lld_identity", keyData.LldIdentity, reason) ||
      !requireEqualInt(obj, "kernel_count",
                       static_cast<int64_t>(keyData.KernelNames.size()),
                       reason) ||
      !validateKernelArray(obj, keyData.KernelNames, reason) ||
      !requireEqualString(obj, "cached_object_sha256", objectSha256, reason) ||
      !requireEqualInt(obj, "cached_object_size",
                       static_cast<int64_t>(objectSize), reason))
    return false;

  auto lifted = requireInt(obj, "lifted_count", reason);
  auto total = requireInt(obj, "total_count", reason);
  auto usesScratch = requireBool(obj, "uses_scratch_private_segment", reason);
  auto sourceScratch =
      requireInt(obj, "source_private_segment_fixed_size", reason);
  auto targetScratch =
      requireInt(obj, "target_private_segment_fixed_size", reason);
  auto targetEnable =
      requireBool(obj, "target_enable_private_segment", reason);
  if (!lifted || !total || !usesScratch ||
      !sourceScratch || !targetScratch || !targetEnable)
    return false;

  result.Success = true;
  result.LiftedCount = static_cast<int>(*lifted);
  result.TotalCount = static_cast<int>(*total);
  result.UsesScratchPrivateSegment = *usesScratch;
  result.SourcePrivateSegmentFixedSize =
      static_cast<uint32_t>(*sourceScratch);
  result.TargetPrivateSegmentFixedSize =
      static_cast<uint32_t>(*targetScratch);
  result.TargetEnablePrivateSegment = *targetEnable;
  return true;
}

} // namespace

const char *translationCacheStatusString(TranslationCacheStatus status) {
  switch (status) {
  case TranslationCacheStatus::Disabled:
    return "disabled";
  case TranslationCacheStatus::Bypassed:
    return "bypassed";
  case TranslationCacheStatus::Miss:
    return "miss";
  case TranslationCacheStatus::Hit:
    return "hit";
  case TranslationCacheStatus::Invalid:
    return "invalid";
  case TranslationCacheStatus::WriteSuccess:
    return "write_success";
  case TranslationCacheStatus::WriteFailed:
    return "write_failed";
  }
  return "invalid";
}

std::string sha256Hex(llvm::ArrayRef<uint8_t> data) {
  auto digest = llvm::SHA256::hash(data);
  std::string out;
  llvm::raw_string_ostream os(out);
  for (uint8_t byte : digest)
    os << llvm::format_hex_no_prefix(byte, 2);
  return os.str();
}

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &request) {
  TranslationCacheLookup lookup;
  if (cacheDisabledByPolicy(request))
    return lookup;

  KeyData keyData = buildKeyData(request);
  lookup.Key = keyData.Key;
  if (!keyData.Error.empty()) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = keyData.Error;
    return lookup;
  }
  lookup.MetadataPath = cacheMetadataPath(request, keyData.Key);
  lookup.ObjectPath = cacheObjectPath(request, keyData.Key);

  const bool metadataExists = exists(lookup.MetadataPath);
  const bool objectExists = exists(lookup.ObjectPath);
  if (!metadataExists && !objectExists) {
    lookup.Status = TranslationCacheStatus::Miss;
    lookup.Reason = "entry not present";
    return lookup;
  }
  if (metadataExists != objectExists) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = metadataExists ? "metadata exists without object"
                                   : "object exists without metadata";
    return lookup;
  }

  auto objectBuffer = llvm::MemoryBuffer::getFile(lookup.ObjectPath);
  if (!objectBuffer) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "failed to read cached object: " +
                    objectBuffer.getError().message();
    return lookup;
  }
  llvm::StringRef objectBytes = (*objectBuffer)->getBuffer();
  std::string objectSha = sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(objectBytes.data()),
      objectBytes.size()));

  auto metadataBuffer = llvm::MemoryBuffer::getFile(lookup.MetadataPath);
  if (!metadataBuffer) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "failed to read cache metadata: " +
                    metadataBuffer.getError().message();
    return lookup;
  }
  auto parsed = llvm::json::parse((*metadataBuffer)->getBuffer());
  if (!parsed) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "failed to parse cache metadata: " +
                    llvm::toString(parsed.takeError());
    return lookup;
  }
  const llvm::json::Object *obj = parsed->getAsObject();
  if (!obj) {
    lookup.Status = TranslationCacheStatus::Invalid;
    lookup.Reason = "cache metadata is not a JSON object";
    return lookup;
  }
  if (!validateMetadata(request, keyData, *obj, objectSha, objectBytes.size(),
                        lookup.Result, lookup.Reason)) {
    lookup.Status = TranslationCacheStatus::Invalid;
    return lookup;
  }

  lookup.Result.Hsaco.assign(
      reinterpret_cast<const uint8_t *>(objectBytes.data()),
      reinterpret_cast<const uint8_t *>(objectBytes.data()) +
          objectBytes.size());
  lookup.Status = TranslationCacheStatus::Hit;
  lookup.Reason = "ok";
  return lookup;
}

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &request, const PipelineResult &result) {
  TranslationCacheWrite write;
  if (cacheDisabledByPolicy(request) || request.CacheReadonly)
    return write;

  KeyData keyData = buildKeyData(request);
  write.Key = keyData.Key;
  if (!keyData.Error.empty()) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = keyData.Error;
    return write;
  }
  write.MetadataPath = cacheMetadataPath(request, keyData.Key);
  write.ObjectPath = cacheObjectPath(request, keyData.Key);

  if (!result.Success || result.Hsaco.empty()) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "refusing to cache unsuccessful or empty translation";
    return write;
  }

  std::string dir = cacheSubdir(request, keyData.Key);
  if (auto ec = llvm::sys::fs::create_directories(dir)) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "failed to create cache directory '" + dir + "': " +
                   ec.message();
    return write;
  }

  std::string objectSha = sha256Hex(result.Hsaco);
  std::string error;
  if (!writeFileAtomic(write.ObjectPath, result.Hsaco, error)) {
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "failed to write cached object: " + error;
    return write;
  }

  llvm::json::Object meta =
      metadataObject(request, keyData, result, objectSha);
  if (!writeFileAtomic(write.MetadataPath,
                       jsonToString(llvm::json::Value(std::move(meta))),
                       error)) {
    llvm::sys::fs::remove(write.ObjectPath);
    write.Status = TranslationCacheStatus::WriteFailed;
    write.Reason = "failed to write cache metadata: " + error;
    return write;
  }

  write.Status = TranslationCacheStatus::WriteSuccess;
  write.Reason = "ok";
  return write;
}

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> kernelNames, llvm::StringRef skipList) {
  if (skipList.empty())
    return "";

  llvm::StringRef remaining(skipList);
  while (!remaining.empty()) {
    auto split = remaining.split(',');
    llvm::StringRef requested = split.first.trim();
    remaining = split.second;
    if (requested.empty())
      continue;
    for (llvm::StringRef kernelName : kernelNames) {
      if (requested == kernelName)
        return kernelName.str();
    }
  }
  return "";
}

} // namespace COMGR::hotswap
