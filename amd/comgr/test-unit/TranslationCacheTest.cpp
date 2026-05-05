//===- translation_cache_test.cpp - translation_cache unit tests ----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <gtest/gtest.h>

#include "hotswap/translation_cache.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TempDir {
  llvm::SmallString<128> Path;
  bool Valid = false;

  explicit TempDir(const char *Prefix) {
    std::error_code Ec = llvm::sys::fs::createUniqueDirectory(Prefix, Path);
    Valid = !Ec;
  }

  ~TempDir() {
    if (Valid)
      llvm::sys::fs::remove_directories(Path);
  }

  std::string file(const char *Name) const {
    llvm::SmallString<256> P(Path);
    llvm::sys::path::append(P, Name);
    return std::string(P);
  }
};

struct ScopedEnv {
  std::string Name;
  std::string OldValue;
  bool HadOldValue = false;

  ScopedEnv(const char *Name, const std::string &Value) : Name(Name) {
    if (const char *Old = std::getenv(Name)) {
      OldValue = Old;
      HadOldValue = true;
    }
    setenv(Name, Value.c_str(), 1);
  }

  ~ScopedEnv() {
    if (HadOldValue)
      setenv(Name.c_str(), OldValue.c_str(), 1);
    else
      unsetenv(Name.c_str());
  }
};

std::vector<uint8_t> fakeAmdgpuElf() {
  std::vector<uint8_t> Data(128, 0);
  Data[0] = 0x7f;
  Data[1] = 'E';
  Data[2] = 'L';
  Data[3] = 'F';
  Data[4] = 2; // ELFCLASS64
  Data[5] = 1; // little-endian
  Data[6] = 1; // current ELF version
  const uint16_t Machine = 224; // EM_AMDGPU
  const uint32_t Flags = 0x49;
  std::memcpy(Data.data() + 18, &Machine, sizeof(Machine));
  std::memcpy(Data.data() + 48, &Flags, sizeof(Flags));
  return Data;
}

void writeTextFile(const std::string &Path, llvm::StringRef Text) {
  std::error_code Ec;
  llvm::raw_fd_ostream Os(Path, Ec);
  ASSERT_FALSE(Ec) << "cannot write " << Path << ": " << Ec.message();
  Os << Text;
}

void writeBinaryFile(const std::string &Path,
                     const std::vector<uint8_t> &Bytes) {
  std::error_code Ec;
  llvm::raw_fd_ostream Os(Path, Ec, llvm::sys::fs::OF_None);
  ASSERT_FALSE(Ec) << "cannot write " << Path << ": " << Ec.message();
  Os.write(reinterpret_cast<const char *>(Bytes.data()), Bytes.size());
}

COMGR::hotswap::TranslationCacheRequest makeRequest(
    const std::vector<uint8_t> &Source, const std::string &RulesPath,
    const std::string &SourceGfx = "gfx1250",
    const std::string &TargetGfx = "gfx942") {
  COMGR::hotswap::TranslationCacheRequest Request;
  Request.SourceObject = llvm::ArrayRef<uint8_t>(Source);
  Request.SourceGfx = SourceGfx;
  Request.TargetGfx = TargetGfx;
  Request.SourceIsa = "amdgcn-amd-amdhsa--" + SourceGfx;
  Request.TargetIsa = "amdgcn-amd-amdhsa--" + TargetGfx;
  Request.CodeIsa = "amdgcn-amd-amdhsa--gfx942";
  Request.HotswapRulesPath = RulesPath;
  Request.OrigMach = 0x49;
  Request.EnableWritelaneRewrite = true;
  Request.EnableWaveNative = true;
  Request.StrictMode = true;
  return Request;
}

COMGR::hotswap::PipelineResult makeSuccessfulResult(
    std::vector<uint8_t> Hsaco = {0x7f, 'E', 'L', 'F', 1, 2, 3}) {
  COMGR::hotswap::PipelineResult Result;
  Result.Success = true;
  Result.Hsaco = std::move(Hsaco);
  Result.LiftedCount = 7;
  Result.TotalCount = 7;
  return Result;
}

} // namespace

TEST(TranslationCache, FirstRunMissWriteSecondRunHit) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(Source, Rules);

  auto First = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(First.Status, COMGR::hotswap::TranslationCacheStatus::Miss);

  auto Result = makeSuccessfulResult();
  auto Write = COMGR::hotswap::writeTranslationCache(Request, Result);
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess)
      << Write.Reason;

  auto Second = COMGR::hotswap::lookupTranslationCache(Request);
  ASSERT_EQ(Second.Status, COMGR::hotswap::TranslationCacheStatus::Hit)
      << Second.Reason;
  EXPECT_EQ(Second.Result.Hsaco, Result.Hsaco);
  EXPECT_EQ(Second.Result.LiftedCount, Result.LiftedCount);
  EXPECT_EQ(Second.Result.TotalCount, Result.TotalCount);
}

TEST(TranslationCache, ChangedInputHashCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(Source, Rules);
  ASSERT_EQ(COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult()).Status,
            COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  Source[80] ^= 0x1;
  auto Changed = makeRequest(Source, Rules);
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Changed);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, ChangedIsaCausesMiss) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(Source, Rules);
  ASSERT_EQ(COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult()).Status,
            COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  auto ChangedSourceIsa = makeRequest(Source, Rules, "gfx1200", "gfx942");
  EXPECT_EQ(COMGR::hotswap::lookupTranslationCache(ChangedSourceIsa).Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);

  auto ChangedTargetIsa = makeRequest(Source, Rules, "gfx1250", "gfx950");
  EXPECT_EQ(COMGR::hotswap::lookupTranslationCache(ChangedTargetIsa).Status,
            COMGR::hotswap::TranslationCacheStatus::Miss);
}

TEST(TranslationCache, CorruptMetadataIsInvalid) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(Source, Rules);
  auto Write = COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  writeTextFile(Write.MetadataPath, "not-json\n");
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Invalid);
  EXPECT_NE(Lookup.Reason.find("parse"), std::string::npos);
}

TEST(TranslationCache, CorruptObjectIsInvalid) {
  TempDir Temp("hotswap_cache_test");
  ASSERT_TRUE(Temp.Valid);
  ScopedEnv CacheDir("HSA_HOTSWAP_CACHE_DIR", Temp.Path.str().str());
  ScopedEnv NoDisable("HSA_HOTSWAP_CACHE_DISABLE", "0");
  ScopedEnv NoReadonly("HSA_HOTSWAP_CACHE_READONLY", "0");

  std::string Rules = Temp.file("rules.json");
  writeTextFile(Rules, "{\"version\":1,\"rules\":[]}\n");
  auto Source = fakeAmdgpuElf();
  auto Request = makeRequest(Source, Rules);
  auto Write = COMGR::hotswap::writeTranslationCache(Request, makeSuccessfulResult());
  ASSERT_EQ(Write.Status, COMGR::hotswap::TranslationCacheStatus::WriteSuccess);

  writeBinaryFile(Write.ObjectPath, {1, 2, 3, 4});
  auto Lookup = COMGR::hotswap::lookupTranslationCache(Request);
  EXPECT_EQ(Lookup.Status, COMGR::hotswap::TranslationCacheStatus::Invalid);
  EXPECT_NE(Lookup.Reason.find("cached_object_sha256"), std::string::npos);
}

TEST(TranslationCache, SkipKernelListMatchesExactKernelName) {
  ScopedEnv Skip("HSA_HOTSWAP_CACHE_SKIP_KERNELS",
                 "other_kernel, target_kernel ,third_kernel");

  std::vector<std::string> Kernels = {"first_kernel", "target_kernel"};
  EXPECT_EQ(COMGR::hotswap::skippedKernelForTranslationCache(Kernels),
            "target_kernel");
}

TEST(TranslationCache, SkipKernelListDoesNotUseSubstringMatching) {
  ScopedEnv Skip("HSA_HOTSWAP_CACHE_SKIP_KERNELS", "target");

  std::vector<std::string> Kernels = {"target_kernel"};
  EXPECT_TRUE(COMGR::hotswap::skippedKernelForTranslationCache(Kernels).empty());
}
