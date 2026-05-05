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
  std::string Path;
  bool Present = false;
  uint64_t Size = 0;
  int64_t MtimeSec = 0;
  int64_t MtimeNsec = 0;
  std::string Sha256;
  std::string Error;
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

std::string envString(const char *Name) {
  const char *Value = std::getenv(Name);
  return Value ? std::string(Value) : std::string();
}

bool envEnabled(const char *Name) {
  const char *Value = std::getenv(Name);
  return Value && Value[0] && std::strcmp(Value, "0") != 0;
}

std::string hexU32(uint32_t Value) {
  std::string Out;
  llvm::raw_string_ostream Os(Out);
  Os << "0x" << llvm::format_hex_no_prefix(Value, 0);
  return Os.str();
}

bool readElfHeaderFields(llvm::ArrayRef<uint8_t> Data, uint16_t &Machine,
                         uint32_t &Flags) {
  auto Buf = llvm::MemoryBuffer::getMemBuffer(
      llvm::StringRef(reinterpret_cast<const char *>(Data.data()),
                      Data.size()),
      "", false);
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(*Buf);
  if (!ObjOrErr) {
    (void)llvm::toString(ObjOrErr.takeError());
    return false;
  }
  const auto *Elf =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(ObjOrErr->get());
  if (!Elf)
    return false;
  Machine = Elf->getEMachine();
  Flags = Elf->getPlatformFlags();
  return true;
}

std::string hashFile(llvm::StringRef Path, std::string &Error) {
  auto Buffer = llvm::MemoryBuffer::getFile(Path);
  if (!Buffer) {
    Error = Buffer.getError().message();
    return "";
  }
  llvm::StringRef Contents = (*Buffer)->getBuffer();
  return sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(Contents.data()), Contents.size()));
}

FileIdentity statIdentity(llvm::StringRef Path) {
  FileIdentity Id;
  Id.Path = Path.str();
  llvm::sys::fs::file_status St;
  if (llvm::sys::fs::status(Path, St))
    return Id;
  if (!llvm::sys::fs::exists(St))
    return Id;
  Id.Present = true;
  Id.Size = static_cast<uint64_t>(St.getSize());
  // TimePoint<> is std::chrono::time_point<system_clock, nanoseconds> on every
  // platform, so split into seconds + nanoseconds-of-second once.
  auto SinceEpoch = St.getLastModificationTime().time_since_epoch();
  auto Secs = std::chrono::duration_cast<std::chrono::seconds>(SinceEpoch);
  Id.MtimeSec = Secs.count();
  Id.MtimeNsec =
      std::chrono::duration_cast<std::chrono::nanoseconds>(SinceEpoch - Secs)
          .count();
  Id.Sha256 = hashFile(Id.Path, Id.Error);
  return Id;
}

std::string identityString(const FileIdentity &Id) {
  std::string Out;
  llvm::raw_string_ostream Os(Out);
  Os << Id.Path << "|present=" << (Id.Present ? "1" : "0")
     << "|size=" << Id.Size << "|mtime=" << Id.MtimeSec << "."
     << Id.MtimeNsec << "|sha256=" << Id.Sha256;
  if (!Id.Error.empty())
    Os << "|error=" << Id.Error;
  return Os.str();
}

std::string loadedImageIdentity() {
  static const std::string identity = [] {
    std::string Out;
    llvm::raw_string_ostream Os(Out);
    Os << "llvm=" << LLVM_VERSION_STRING;
    // getMainExecutable uses dladdr/Mach-O lookup on Unix and
    // GetModuleFileName on Windows; either way we get the filesystem path
    // of the loaded image so its mtime/sha256 can key the cache.
    std::string Image = llvm::sys::fs::getMainExecutable(
        nullptr, reinterpret_cast<void *>(&loadedImageIdentity));
    if (!Image.empty())
      Os << "|image=" << identityString(statIdentity(Image));
    else
      Os << "|image=<unavailable>";
    return Os.str();
  }();
  return identity;
}

void appendKeyField(std::string &Material, llvm::StringRef Name,
                    llvm::StringRef Value) {
  Material.append(Name.data(), Name.size());
  Material.push_back('\0');
  Material += std::to_string(Value.size());
  Material.push_back(':');
  if (!Value.empty())
    Material.append(Value.data(), Value.size());
  Material.push_back('\0');
}

void appendKeyField(std::string &Material, llvm::StringRef Name, bool Value) {
  appendKeyField(Material, Name, llvm::StringRef(Value ? "true" : "false"));
}

void appendKeyField(std::string &Material, llvm::StringRef Name, int Value) {
  appendKeyField(Material, Name, std::to_string(Value));
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

KeyData buildKeyData(const TranslationCacheRequest &Request) {
  KeyData Data;
  if (Request.SourceObject.empty()) {
    Data.Error = "empty source code object";
    return Data;
  }
  if (Request.SourceGfx.empty() || Request.TargetGfx.empty()) {
    Data.Error = "missing source or target gfx";
    return Data;
  }

  Data.SourceSha256 = sha256Hex(Request.SourceObject);
  uint16_t Machine = 0;
  uint32_t Flags = 0;
  if (readElfHeaderFields(Request.SourceObject, Machine, Flags)) {
    Data.ElfMachineHex = hexU32(Machine);
    Data.ElfFlagsHex = hexU32(Flags);
  } else {
    Data.Error = "source code object is not a 64-bit little-endian ELF";
    return Data;
  }

  if (!Request.HotswapRulesPath.empty()) {
    Data.RulesSha256 = hashFile(Request.HotswapRulesPath, Data.RulesError);
    if (!Data.RulesError.empty()) {
      Data.Error = "failed to hash HSA_HOTSWAP_RULES '" +
                   Request.HotswapRulesPath + "': " + Data.RulesError;
      return Data;
    }
  }

  const std::string toolsDir = LLVM_TOOLS_DIR;
  const FileIdentity &Llc = llcIdentity();
  const FileIdentity &LlvmMc = llvmMcIdentity();
  const FileIdentity &Lld = lldIdentity();
  if (!Llc.Present || !LlvmMc.Present || !Lld.Present || !Llc.Error.empty() ||
      !LlvmMc.Error.empty() || !Lld.Error.empty()) {
    Data.Error = "LLVM tool identity is incomplete under " + toolsDir;
    return Data;
  }
  Data.LlcIdentity = identityString(Llc);
  Data.LlvmMcIdentity = identityString(LlvmMc);
  Data.LldIdentity = identityString(Lld);
  Data.BuildIdentity = loadedImageIdentity();
  Data.KernelNames = listKernelNames(
      std::vector<uint8_t>(Request.SourceObject.begin(),
                           Request.SourceObject.end()));

  std::string Material;
  appendKeyField(Material, "schema", std::to_string(kCacheSchemaVersion));
  appendKeyField(Material, "source_sha256", Data.SourceSha256);
  appendKeyField(Material, "source_gfx", Request.SourceGfx);
  appendKeyField(Material, "target_gfx", Request.TargetGfx);
  appendKeyField(Material, "source_isa", Request.SourceIsa);
  appendKeyField(Material, "target_isa", Request.TargetIsa);
  appendKeyField(Material, "code_isa", Request.CodeIsa);
  appendKeyField(Material, "elf_machine", Data.ElfMachineHex);
  appendKeyField(Material, "elf_flags", Data.ElfFlagsHex);
  appendKeyField(Material, "orig_mach", Request.OrigMach);
  appendKeyField(Material, "rules_path", Request.HotswapRulesPath);
  appendKeyField(Material, "rules_sha256", Data.RulesSha256);
  appendKeyField(Material, "strict", Request.StrictMode);
  appendKeyField(Material, "enable_writelane_rewrite",
                 Request.EnableWritelaneRewrite);
  appendKeyField(Material, "enable_wave_native", Request.EnableWaveNative);
  appendKeyField(Material, "hotswap_build_identity", Data.BuildIdentity);
  appendKeyField(Material, "llc_identity", Data.LlcIdentity);
  appendKeyField(Material, "llvm_mc_identity", Data.LlvmMcIdentity);
  appendKeyField(Material, "lld_identity", Data.LldIdentity);
  Data.Key = sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(Material.data()), Material.size()));
  return Data;
}

std::string cacheRoot() { return envString("HSA_HOTSWAP_CACHE_DIR"); }

bool cacheDisabledByEnv() {
  return envEnabled("HSA_HOTSWAP_CACHE_DISABLE") || cacheRoot().empty();
}

std::string cacheSubdir(llvm::StringRef Key) {
  llvm::SmallString<256> Path(cacheRoot());
  llvm::sys::path::append(Path, Key.substr(0, 2));
  return std::string(Path);
}

std::string cacheObjectPath(llvm::StringRef Key) {
  llvm::SmallString<256> Path(cacheSubdir(Key));
  llvm::sys::path::append(Path, llvm::Twine(Key) + ".hsaco");
  return std::string(Path);
}

std::string cacheMetadataPath(llvm::StringRef Key) {
  llvm::SmallString<256> Path(cacheSubdir(Key));
  llvm::sys::path::append(Path, llvm::Twine(Key) + ".json");
  return std::string(Path);
}

bool exists(llvm::StringRef Path) {
  return llvm::sys::fs::exists(Path);
}

std::string jsonToString(llvm::json::Value Value) {
  std::string Out;
  llvm::raw_string_ostream Os(Out);
  Value.print(Os);
  Os << "\n";
  return Os.str();
}

bool writeFileAtomic(llvm::StringRef Path, llvm::StringRef Contents,
                     std::string &Error) {
  llvm::SmallString<256> Model(Path);
  Model += ".tmp-%%%%%%";
  llvm::SmallString<256> TmpPath;
  int Fd = -1;
  if (auto Ec = llvm::sys::fs::createUniqueFile(Model, Fd, TmpPath)) {
    Error = Ec.message();
    return false;
  }
  {
    llvm::raw_fd_ostream Os(Fd, true);
    Os << Contents;
    if (Os.has_error()) {
      Error = Os.error().message();
      Os.clear_error();
      llvm::sys::fs::remove(TmpPath);
      return false;
    }
  }
  if (auto Ec = llvm::sys::fs::rename(TmpPath, Path)) {
    Error = Ec.message();
    llvm::sys::fs::remove(TmpPath);
    return false;
  }
  return true;
}

bool writeFileAtomic(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data,
                     std::string &Error) {
  return writeFileAtomic(
      Path, llvm::StringRef(reinterpret_cast<const char *>(Data.data()),
                            Data.size()),
      Error);
}

std::optional<std::string> requireString(const llvm::json::Object &Obj,
                                         llvm::StringRef Field,
                                         std::string &Reason) {
  auto Value = Obj.getString(Field);
  if (!Value) {
    Reason = "metadata field '" + Field.str() + "' missing or not a string";
    return std::nullopt;
  }
  return Value->str();
}

std::optional<int64_t> requireInt(const llvm::json::Object &Obj,
                                  llvm::StringRef Field, std::string &Reason) {
  auto Value = Obj.getInteger(Field);
  if (!Value) {
    Reason = "metadata field '" + Field.str() + "' missing or not an integer";
    return std::nullopt;
  }
  return *Value;
}

std::optional<bool> requireBool(const llvm::json::Object &Obj,
                                llvm::StringRef Field, std::string &Reason) {
  auto Value = Obj.getBoolean(Field);
  if (!Value) {
    Reason = "metadata field '" + Field.str() + "' missing or not a boolean";
    return std::nullopt;
  }
  return *Value;
}

bool requireEqualString(const llvm::json::Object &Obj, llvm::StringRef Field,
                        llvm::StringRef Expected, std::string &Reason) {
  auto Value = requireString(Obj, Field, Reason);
  if (!Value)
    return false;
  if (*Value != Expected) {
    Reason = "metadata field '" + Field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualInt(const llvm::json::Object &Obj, llvm::StringRef Field,
                     int64_t Expected, std::string &Reason) {
  auto Value = requireInt(Obj, Field, Reason);
  if (!Value)
    return false;
  if (*Value != Expected) {
    Reason = "metadata field '" + Field.str() + "' mismatch";
    return false;
  }
  return true;
}

bool requireEqualBool(const llvm::json::Object &Obj, llvm::StringRef Field,
                      bool Expected, std::string &Reason) {
  auto Value = requireBool(Obj, Field, Reason);
  if (!Value)
    return false;
  if (*Value != Expected) {
    Reason = "metadata field '" + Field.str() + "' mismatch";
    return false;
  }
  return true;
}

llvm::json::Array kernelArray(const std::vector<std::string> &KernelNames) {
  llvm::json::Array Arr;
  for (llvm::StringRef Name : KernelNames)
    Arr.push_back(Name);
  return Arr;
}

bool validateKernelArray(const llvm::json::Object &Obj,
                         const std::vector<std::string> &Expected,
                         std::string &Reason) {
  const llvm::json::Array *Arr = Obj.getArray("kernel_names");
  if (!Arr) {
    Reason = "metadata field 'kernel_names' missing or not an array";
    return false;
  }
  if (Arr->size() != Expected.size()) {
    Reason = "metadata kernel_names size mismatch";
    return false;
  }
  for (size_t I = 0; I < Expected.size(); ++I) {
    auto Value = (*Arr)[I].getAsString();
    if (!Value || *Value != Expected[I]) {
      Reason = "metadata kernel_names mismatch";
      return false;
    }
  }
  return true;
}

llvm::json::Object metadataObject(const TranslationCacheRequest &Request,
                                  const KeyData &KeyData,
                                  const PipelineResult &Result,
                                  llvm::StringRef ObjectSha256) {
  return llvm::json::Object{
      {"schema_version", kCacheSchemaVersion},
      {"key", KeyData.Key},
      {"source_object_sha256", KeyData.SourceSha256},
      {"source_gfx", Request.SourceGfx},
      {"target_gfx", Request.TargetGfx},
      {"source_isa", Request.SourceIsa},
      {"target_isa", Request.TargetIsa},
      {"code_isa", Request.CodeIsa},
      {"elf_machine", KeyData.ElfMachineHex},
      {"elf_flags", KeyData.ElfFlagsHex},
      {"orig_mach", Request.OrigMach},
      {"hotswap_rules_path", Request.HotswapRulesPath},
      {"hotswap_rules_sha256", KeyData.RulesSha256},
      {"strict_mode", Request.StrictMode},
      {"enable_writelane_rewrite", Request.EnableWritelaneRewrite},
      {"enable_wave_native", Request.EnableWaveNative},
      {"hotswap_build_identity", KeyData.BuildIdentity},
      {"llc_identity", KeyData.LlcIdentity},
      {"llvm_mc_identity", KeyData.LlvmMcIdentity},
      {"lld_identity", KeyData.LldIdentity},
      {"kernel_count", static_cast<int64_t>(KeyData.KernelNames.size())},
      {"kernel_names", kernelArray(KeyData.KernelNames)},
      {"cached_object_sha256", ObjectSha256.str()},
      {"cached_object_size", static_cast<int64_t>(Result.Hsaco.size())},
      {"lifted_count", Result.LiftedCount},
      {"total_count", Result.TotalCount},
      {"uses_scratch_private_segment", Result.UsesScratchPrivateSegment},
      {"source_private_segment_fixed_size",
       static_cast<int64_t>(Result.SourcePrivateSegmentFixedSize)},
      {"target_private_segment_fixed_size",
       static_cast<int64_t>(Result.TargetPrivateSegmentFixedSize)},
      {"target_enable_private_segment", Result.TargetEnablePrivateSegment},
  };
}

bool validateMetadata(const TranslationCacheRequest &Request,
                      const KeyData &KeyData, const llvm::json::Object &Obj,
                      llvm::StringRef ObjectSha256, size_t ObjectSize,
                      PipelineResult &Result, std::string &Reason) {
  if (!requireEqualInt(Obj, "schema_version", kCacheSchemaVersion, Reason) ||
      !requireEqualString(Obj, "key", KeyData.Key, Reason) ||
      !requireEqualString(Obj, "source_object_sha256", KeyData.SourceSha256,
                          Reason) ||
      !requireEqualString(Obj, "source_gfx", Request.SourceGfx, Reason) ||
      !requireEqualString(Obj, "target_gfx", Request.TargetGfx, Reason) ||
      !requireEqualString(Obj, "source_isa", Request.SourceIsa, Reason) ||
      !requireEqualString(Obj, "target_isa", Request.TargetIsa, Reason) ||
      !requireEqualString(Obj, "code_isa", Request.CodeIsa, Reason) ||
      !requireEqualString(Obj, "elf_machine", KeyData.ElfMachineHex, Reason) ||
      !requireEqualString(Obj, "elf_flags", KeyData.ElfFlagsHex, Reason) ||
      !requireEqualInt(Obj, "orig_mach", Request.OrigMach, Reason) ||
      !requireEqualString(Obj, "hotswap_rules_path", Request.HotswapRulesPath,
                          Reason) ||
      !requireEqualString(Obj, "hotswap_rules_sha256", KeyData.RulesSha256,
                          Reason) ||
      !requireEqualBool(Obj, "strict_mode", Request.StrictMode, Reason) ||
      !requireEqualBool(Obj, "enable_writelane_rewrite",
                        Request.EnableWritelaneRewrite, Reason) ||
      !requireEqualBool(Obj, "enable_wave_native", Request.EnableWaveNative,
                        Reason) ||
      !requireEqualString(Obj, "hotswap_build_identity",
                          KeyData.BuildIdentity, Reason) ||
      !requireEqualString(Obj, "llc_identity", KeyData.LlcIdentity, Reason) ||
      !requireEqualString(Obj, "llvm_mc_identity", KeyData.LlvmMcIdentity,
                          Reason) ||
      !requireEqualString(Obj, "lld_identity", KeyData.LldIdentity, Reason) ||
      !requireEqualInt(Obj, "kernel_count",
                       static_cast<int64_t>(KeyData.KernelNames.size()),
                       Reason) ||
      !validateKernelArray(Obj, KeyData.KernelNames, Reason) ||
      !requireEqualString(Obj, "cached_object_sha256", ObjectSha256, Reason) ||
      !requireEqualInt(Obj, "cached_object_size",
                       static_cast<int64_t>(ObjectSize), Reason))
    return false;

  auto Lifted = requireInt(Obj, "lifted_count", Reason);
  auto Total = requireInt(Obj, "total_count", Reason);
  auto UsesScratch = requireBool(Obj, "uses_scratch_private_segment", Reason);
  auto SourceScratch =
      requireInt(Obj, "source_private_segment_fixed_size", Reason);
  auto TargetScratch =
      requireInt(Obj, "target_private_segment_fixed_size", Reason);
  auto TargetEnable =
      requireBool(Obj, "target_enable_private_segment", Reason);
  if (!Lifted || !Total || !UsesScratch ||
      !SourceScratch || !TargetScratch || !TargetEnable)
    return false;

  Result.Success = true;
  Result.LiftedCount = static_cast<int>(*Lifted);
  Result.TotalCount = static_cast<int>(*Total);
  Result.UsesScratchPrivateSegment = *UsesScratch;
  Result.SourcePrivateSegmentFixedSize =
      static_cast<uint32_t>(*SourceScratch);
  Result.TargetPrivateSegmentFixedSize =
      static_cast<uint32_t>(*TargetScratch);
  Result.TargetEnablePrivateSegment = *TargetEnable;
  return true;
}

} // namespace

const char *translationCacheStatusString(TranslationCacheStatus Status) {
  switch (Status) {
  case TranslationCacheStatus::Disabled:
    return "disabled";
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

std::string sha256Hex(llvm::ArrayRef<uint8_t> Data) {
  auto Digest = llvm::SHA256::hash(Data);
  std::string Out;
  llvm::raw_string_ostream Os(Out);
  for (uint8_t Byte : Digest)
    Os << llvm::format_hex_no_prefix(Byte, 2);
  return Os.str();
}

TranslationCacheLookup lookupTranslationCache(
    const TranslationCacheRequest &Request) {
  TranslationCacheLookup Lookup;
  if (cacheDisabledByEnv())
    return Lookup;

  KeyData KeyData = buildKeyData(Request);
  Lookup.Key = KeyData.Key;
  if (!KeyData.Error.empty()) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    Lookup.Reason = KeyData.Error;
    return Lookup;
  }
  Lookup.MetadataPath = cacheMetadataPath(KeyData.Key);
  Lookup.ObjectPath = cacheObjectPath(KeyData.Key);

  const bool metadataExists = exists(Lookup.MetadataPath);
  const bool objectExists = exists(Lookup.ObjectPath);
  if (!metadataExists && !objectExists) {
    Lookup.Status = TranslationCacheStatus::Miss;
    Lookup.Reason = "entry not present";
    return Lookup;
  }
  if (metadataExists != objectExists) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    Lookup.Reason = metadataExists ? "metadata exists without object"
                                   : "object exists without metadata";
    return Lookup;
  }

  auto ObjectBuffer = llvm::MemoryBuffer::getFile(Lookup.ObjectPath);
  if (!ObjectBuffer) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    Lookup.Reason = "failed to read cached object: " +
                    ObjectBuffer.getError().message();
    return Lookup;
  }
  llvm::StringRef ObjectBytes = (*ObjectBuffer)->getBuffer();
  std::string ObjectSha = sha256Hex(llvm::ArrayRef<uint8_t>(
      reinterpret_cast<const uint8_t *>(ObjectBytes.data()),
      ObjectBytes.size()));

  auto MetadataBuffer = llvm::MemoryBuffer::getFile(Lookup.MetadataPath);
  if (!MetadataBuffer) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    Lookup.Reason = "failed to read cache metadata: " +
                    MetadataBuffer.getError().message();
    return Lookup;
  }
  auto Parsed = llvm::json::parse((*MetadataBuffer)->getBuffer());
  if (!Parsed) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    Lookup.Reason = "failed to parse cache metadata: " +
                    llvm::toString(Parsed.takeError());
    return Lookup;
  }
  const llvm::json::Object *Obj = Parsed->getAsObject();
  if (!Obj) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    Lookup.Reason = "cache metadata is not a JSON object";
    return Lookup;
  }
  if (!validateMetadata(Request, KeyData, *Obj, ObjectSha, ObjectBytes.size(),
                        Lookup.Result, Lookup.Reason)) {
    Lookup.Status = TranslationCacheStatus::Invalid;
    return Lookup;
  }

  Lookup.Result.Hsaco.assign(
      reinterpret_cast<const uint8_t *>(ObjectBytes.data()),
      reinterpret_cast<const uint8_t *>(ObjectBytes.data()) +
          ObjectBytes.size());
  Lookup.Status = TranslationCacheStatus::Hit;
  Lookup.Reason = "ok";
  return Lookup;
}

TranslationCacheWrite writeTranslationCache(
    const TranslationCacheRequest &Request, const PipelineResult &Result) {
  TranslationCacheWrite Write;
  if (cacheDisabledByEnv() || envEnabled("HSA_HOTSWAP_CACHE_READONLY"))
    return Write;

  KeyData KeyData = buildKeyData(Request);
  Write.Key = KeyData.Key;
  if (!KeyData.Error.empty()) {
    Write.Status = TranslationCacheStatus::WriteFailed;
    Write.Reason = KeyData.Error;
    return Write;
  }
  Write.MetadataPath = cacheMetadataPath(KeyData.Key);
  Write.ObjectPath = cacheObjectPath(KeyData.Key);

  if (!Result.Success || Result.Hsaco.empty()) {
    Write.Status = TranslationCacheStatus::WriteFailed;
    Write.Reason = "refusing to cache unsuccessful or empty translation";
    return Write;
  }

  std::string Dir = cacheSubdir(KeyData.Key);
  if (auto Ec = llvm::sys::fs::create_directories(Dir)) {
    Write.Status = TranslationCacheStatus::WriteFailed;
    Write.Reason = "failed to create cache directory '" + Dir + "': " +
                   Ec.message();
    return Write;
  }

  std::string ObjectSha = sha256Hex(Result.Hsaco);
  std::string Error;
  if (!writeFileAtomic(Write.ObjectPath, Result.Hsaco, Error)) {
    Write.Status = TranslationCacheStatus::WriteFailed;
    Write.Reason = "failed to write cached object: " + Error;
    return Write;
  }

  llvm::json::Object Meta =
      metadataObject(Request, KeyData, Result, ObjectSha);
  if (!writeFileAtomic(Write.MetadataPath,
                       jsonToString(llvm::json::Value(std::move(Meta))),
                       Error)) {
    llvm::sys::fs::remove(Write.ObjectPath);
    Write.Status = TranslationCacheStatus::WriteFailed;
    Write.Reason = "failed to write cache metadata: " + Error;
    return Write;
  }

  Write.Status = TranslationCacheStatus::WriteSuccess;
  Write.Reason = "ok";
  return Write;
}

std::string skippedKernelForTranslationCache(
    llvm::ArrayRef<std::string> KernelNames) {
  const char *SkipEnv = std::getenv("HSA_HOTSWAP_CACHE_SKIP_KERNELS");
  if (!SkipEnv || !SkipEnv[0])
    return "";

  llvm::StringRef Remaining(SkipEnv);
  while (!Remaining.empty()) {
    auto Split = Remaining.split(',');
    llvm::StringRef Requested = Split.first.trim();
    Remaining = Split.second;
    if (Requested.empty())
      continue;
    for (llvm::StringRef KernelName : KernelNames) {
      if (Requested == KernelName)
        return KernelName.str();
    }
  }
  return "";
}

} // namespace COMGR::hotswap
