//===- pipeline.cpp - Hotswap transpiler ----------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pipeline.h"
#include "code_object_utils.h"
#include "raiser.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/StringExtras.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#define DEBUG_TYPE "transpiler"

#ifndef LLVM_TOOLS_DIR
#define LLVM_TOOLS_DIR "/usr/bin"
#endif

namespace COMGR::hotswap {

namespace {

bool writeFile(llvm::StringRef Path, llvm::StringRef Contents) {
  std::ofstream F(Path.str());
  if (!F.is_open()) {
    llvm::errs() << "transpiler: Cannot write file: " << Path << "\n";
    return false;
  }
  F.write(Contents.data(), Contents.size());
  F.flush();
  if (!F) {
    llvm::errs() << "transpiler: write failed for: " << Path << "\n";
    return false;
  }
  return true;
}

bool writeFile(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data) {
  std::ofstream F(Path.str(), std::ios::binary);
  if (!F.is_open()) {
    llvm::errs() << "transpiler: Cannot write file: " << Path << "\n";
    return false;
  }
  F.write(reinterpret_cast<const char *>(Data.data()), Data.size());
  F.flush();
  if (!F) {
    llvm::errs() << "transpiler: write failed for: " << Path << "\n";
    return false;
  }
  return true;
}

// Derive a filesystem-safe basename for an arbitrarily long kernel name.
// Most POSIX filesystems cap individual path components at 255 bytes, and
// Hotswap generates sibling files off the same stem (e.g. `<stem>.ll`,
// `<stem>.s`, `<stem>.dis`), so we leave a small suffix budget and fold
// anything longer down to a deterministic truncated+hashed form so two
// kernels with a shared 240-byte prefix don't collide on disk.
//
// The returned basename preserves a readable prefix of the original name
// for debuggability; it's only intended for temp-dir scratch files —
// symbol names inside the IR itself are unaffected.
std::string makeSafeBasename(llvm::StringRef KernelName,
                             size_t ReservedSuffixBytes = 8) {
  constexpr size_t kMaxComponentBytes = 255;
  if (KernelName.size() + ReservedSuffixBytes <= kMaxComponentBytes)
    return KernelName.str();

  // FNV-1a 64-bit hash — small, deterministic, no libstdc++ dep beyond cstdint.
  uint64_t H = 0xcbf29ce484222325ull;
  for (unsigned char C : KernelName) {
    H ^= C;
    H *= 0x100000001b3ull;
  }

  constexpr size_t kHashHexBytes = 16;   // "%016llx"
  constexpr size_t kSeparatorBytes = 1;  // '_'
  const size_t prefixBudget = kMaxComponentBytes - ReservedSuffixBytes -
                              kHashHexBytes - kSeparatorBytes;
  std::string Prefix = KernelName.substr(0, prefixBudget).str();
  char Buf[32];
  std::snprintf(Buf, sizeof(Buf), "%016llx",
                static_cast<unsigned long long>(H));
  return Prefix + "_" + Buf;
}

int toolTimeoutSeconds() {
  static const int timeout = [] {
    constexpr int kDefaultTimeoutSeconds = 300;
    const char *Env = std::getenv("HSA_HOTSWAP_TOOL_TIMEOUT_S");
    if (!Env || !Env[0])
      return kDefaultTimeoutSeconds;
    char *End = nullptr;
    long Parsed = std::strtol(Env, &End, 10);
    if (*End != '\0' || Parsed <= 0) {
      llvm::errs() << "transpiler: invalid HSA_HOTSWAP_TOOL_TIMEOUT_S='"
                   << Env << "'; using default " << kDefaultTimeoutSeconds
                   << " seconds\n";
      return kDefaultTimeoutSeconds;
    }
    return static_cast<int>(Parsed);
  }();
  return timeout;
}

int runTool(llvm::StringRef Program, llvm::ArrayRef<llvm::StringRef> Args) {
  LLVM_DEBUG({
    llvm::dbgs() << "transpiler: Running:";
    for (auto &a : Args) llvm::dbgs() << " " << a;
    llvm::dbgs() << "\n";
  });

  auto ExeOrErr = llvm::sys::findProgramByName(Program);
  if (!ExeOrErr) {
    llvm::errs() << "transpiler: tool not found: " << Program << "\n";
    return -1;
  }

  std::string ErrMsg;
  int Rc = llvm::sys::ExecuteAndWait(*ExeOrErr, Args, /*Env=*/std::nullopt,
                                     /*Redirects=*/{},
                                     /*SecondsToWait=*/toolTimeoutSeconds(),
                                     /*MemoryLimit=*/0, &ErrMsg);
  if (Rc != 0)
    llvm::errs() << "transpiler: " << Program << " failed (exit " << Rc << ")"
                 << (ErrMsg.empty() ? "" : ": " + ErrMsg) << "\n";
  return Rc;
}

struct DumpDir {
  llvm::SmallString<128> Path;
  bool Valid = false;
  bool Persistent = false;

  DumpDir() {
    static const char *EnvDir = std::getenv("HSA_HOTSWAP_DUMP_DIR");
    if (EnvDir && EnvDir[0]) {
      Persistent = true;
      Path = EnvDir;
      if (auto Ec = llvm::sys::fs::create_directories(Path)) {
        llvm::errs() << "hotswap: failed to create dump dir '"
                     << Path << "': " << Ec.message() << "\n";
        return;
      }
      // Create a unique subdirectory per invocation so parallel runs
      // don't clobber each other.
      llvm::SmallString<128> Sub;
      if (auto Ec = llvm::sys::fs::createUniqueDirectory(
              Path + "/hotswap", Sub)) {
        llvm::errs() << "hotswap: failed to create subdir in '"
                     << Path << "': " << Ec.message() << "\n";
        return;
      }
      Path = Sub;
      Valid = true;
    } else {
      if (auto Ec =
              llvm::sys::fs::createUniqueDirectory("transpiler", Path)) {
        llvm::errs() << "hotswap: failed to create temp dir: "
                     << Ec.message() << "\n";
      } else {
        Valid = true;
      }
    }
  }

  ~DumpDir() {
    if (Valid && !Persistent)
      llvm::sys::fs::remove_directories(Path);
  }

  DumpDir(const DumpDir &) = delete;
  DumpDir &operator=(const DumpDir &) = delete;

  std::string filePath(llvm::StringRef Name) const {
    llvm::SmallString<256> P(Path);
    llvm::sys::path::append(P, Name);
    return std::string(P);
  }
};

} // anonymous namespace

bool isStrictMode() {
  // Parsed once on first call. The handler implementations call this on
  // every relevant instruction, so going through the OS allocator
  // (`std::getenv`) repeatedly would be wasteful; the result also cannot
  // change inside a process because the env var is read once at the
  // first transpile and reused for the rest of the process lifetime.
  // Treats any non-empty value as enabled to keep the runner side
  // (`HSA_HOTSWAP_STRICT=1`) and the pipeline side decoupled — a future
  // shell that writes `HSA_HOTSWAP_STRICT=true` still works.
  static const bool s_strict = []() {
    const char *V = std::getenv("HSA_HOTSWAP_STRICT");
    return V && V[0] != '\0';
  }();
  return s_strict;
}

// Raise one kernel to IR, compile to a relocatable .o via llc + llvm-mc.
// On success, writes the .o to objPath and returns true.
static bool raiseAndCompileKernel(const TextSection &Text,
                                  llvm::ArrayRef<uint8_t> CodeObjectData,
                                  llvm::StringRef KernelName,
                                  llvm::StringRef SourceIsa,
                                  llvm::StringRef TargetIsa,
                                  const DumpDir &TmpDir,
                                  llvm::StringRef ObjPath,
                                  PipelineResult &Result) {
  auto Meta = extractKernelMeta(CodeObjectData, KernelName);
  if (Meta.Args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << KernelName
                 << "', using empty metadata\n";
  }

  uint64_t KernelOffset = findKernelSymbolOffset(CodeObjectData, KernelName);
  LLVM_DEBUG(if (KernelOffset > 0)
    llvm::dbgs() << "transpiler: Kernel '" << KernelName
                 << "' at .text offset 0x" << llvm::utohexstr(KernelOffset)
                 << "\n");

  auto Raised = raiseToIR(Text.Bytes, SourceIsa, KernelName, Meta, KernelOffset,
                           TargetIsa);
  if (!Raised.Success) {
    llvm::errs() << "transpiler: Raising '" << KernelName << "' to LLVM IR failed";
    Result.FailKernel = KernelName;
    if (!Raised.Failure.Mnemonic.empty()) {
      llvm::errs() << " (unsupported: " << Raised.Failure.Mnemonic << ")";
      Result.FailMnemonic = Raised.Failure.Mnemonic;
    }
    if (Raised.Failure.hasFailed()) {
      Result.FailReason = reasonString(Raised.Failure.Reason);
      Result.FailFormat = Raised.Failure.Format;
      Result.FailDetail = Raised.Failure.Detail;
      Result.FailOffset = Raised.Failure.Offset;
    }
    llvm::errs() << "\n";
    return false;
  }
  Result.LiftedCount += Raised.LiftedCount;
  Result.TotalCount += Raised.TotalCount;
  if (Raised.UsesScratchPrivateSegment) {
    Result.UsesScratchPrivateSegment = true;
    if (Raised.SourcePrivateSegmentFixedSize >
        Result.SourcePrivateSegmentFixedSize)
      Result.SourcePrivateSegmentFixedSize = Raised.SourcePrivateSegmentFixedSize;
  }
  if (!Result.IrText.empty())
    Result.IrText += "\n";
  Result.IrText += Raised.IrText;

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raised '" << KernelName << "' "
                           << Raised.LiftedCount << "/"
                           << Raised.TotalCount << " instructions\n");

  // Kernel names from Tensile et al. routinely exceed 255 bytes, which is
  // the per-component limit on ext4/xfs/tmpfs.  makeSafeBasename() hashes
  // the tail and truncates the head when the full name would blow the
  // budget; the symbol name inside the IR stays untouched, so debug
  // tooling can still resolve the long name from the LLVM module.
  std::string FileStem = makeSafeBasename(KernelName, /*reservedSuffixBytes=*/5);
  std::string IrPath  = TmpDir.filePath(FileStem + ".ll");
  std::string AsmPath = TmpDir.filePath(FileStem + ".s");

  if (!writeFile(IrPath, Raised.IrText))
    return false;

  static const char *SDumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (SDumpInput && SDumpInput[0] == '1' && !Raised.DisasmText.empty())
    writeFile(TmpDir.filePath(FileStem + ".dis"), Raised.DisasmText);

  std::string LlcBin = std::string(LLVM_TOOLS_DIR) + "/llc";
  std::string McpuLlc = ("-mcpu=" + TargetIsa).str();
  if (runTool(LlcBin, {LlcBin, "-march=amdgcn", McpuLlc, "-filetype=asm", "-o",
                       AsmPath, IrPath}) != 0) {
    llvm::errs() << "transpiler: llc failed for '" << KernelName << "'\n";
    return false;
  }

  {
    auto AsmData = readFile(AsmPath);
    if (!Result.AsmText.empty())
      Result.AsmText += "\n";
    Result.AsmText.append(AsmData.begin(), AsmData.end());
  }

  std::string McBin = std::string(LLVM_TOOLS_DIR) + "/llvm-mc";
  std::string McpuMc = ("-mcpu=" + TargetIsa).str();
  if (runTool(McBin, {McBin, "-triple=amdgcn-amd-amdhsa", McpuMc,
                      "-filetype=obj", "-o", ObjPath, AsmPath}) != 0) {
    llvm::errs() << "transpiler: llvm-mc failed for '" << KernelName << "'\n";
    return false;
  }

  return true;
}

// Link one or more relocatable .o files into a shared HSACO.
static bool linkObjects(llvm::ArrayRef<std::string> ObjPaths,
                        llvm::StringRef HsacoPath) {
  std::string LldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  llvm::SmallVector<llvm::StringRef, 16> Args;
  Args.push_back(LldBin);
  Args.push_back("-shared");
  Args.push_back("-o");
  Args.push_back(HsacoPath);
  for (auto &O : ObjPaths)
    Args.push_back(O);
  if (runTool(LldBin, Args) != 0) {
    llvm::errs() << "transpiler: ld.lld failed\n";
    return false;
  }
  return true;
}

void collectTargetPrivateSegmentMetadata(PipelineResult &Result,
                                         llvm::ArrayRef<std::string> KernelNames) {
  using namespace llvm::amdhsa;
  if (Result.Hsaco.empty())
    return;
  for (llvm::StringRef KernelName : KernelNames) {
    KernelMeta Meta = extractKernelMeta(Result.Hsaco, KernelName);
    if (!Meta.HasKernelDescriptor)
      continue;
    Result.TargetPrivateSegmentFixedSize = std::max(
        Result.TargetPrivateSegmentFixedSize,
        Meta.PrivateSegmentFixedSize);
    const bool enabled =
        (Meta.ComputePgmRsrc2 &
         (1u << COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT)) != 0;
    Result.TargetEnablePrivateSegment |= enabled;
  }
}

PipelineResult runPipeline(llvm::ArrayRef<uint8_t> CodeObjectData,
                           llvm::StringRef SourceIsa,
                           llvm::StringRef TargetIsa,
                           llvm::StringRef KernelName) {
  PipelineResult Result;

  auto Text = extractTextSection(CodeObjectData);
  if (!Text.Valid) {
    llvm::errs() << "transpiler: Failed to extract .text section\n";
    return Result;
  }

  DumpDir TmpDir;
  if (!TmpDir.Valid)
    return Result;

  {
    static const char *SDumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (SDumpInput && SDumpInput[0] == '1')
      writeFile(TmpDir.filePath("input.co"), CodeObjectData);
  }

  std::string ObjPath   = TmpDir.filePath("kernel.o");
  std::string HsacoPath = TmpDir.filePath("kernel.hsaco");

  if (!raiseAndCompileKernel(Text, CodeObjectData, KernelName,
                             SourceIsa, TargetIsa, TmpDir, ObjPath, Result))
    return Result;

  if (!linkObjects({ObjPath}, HsacoPath))
    return Result;

  Result.Hsaco = readFile(HsacoPath);
  if (Result.Hsaco.empty()) {
    llvm::errs() << "transpiler: Failed to read HSACO\n";
    return Result;
  }
  std::string KernelNameStr = KernelName.str();
  collectTargetPrivateSegmentMetadata(Result, {KernelNameStr});

  LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: " << Result.Hsaco.size()
                          << " bytes\n");
  Result.Success = true;
  return Result;
}

PipelineResult runPipelineAllKernels(llvm::ArrayRef<uint8_t> CodeObjectData,
                                     llvm::StringRef SourceIsa,
                                     llvm::StringRef TargetIsa) {
  PipelineResult Result;

  auto KernelNames = listKernelNames(CodeObjectData);
  if (KernelNames.empty()) {
    llvm::errs() << "transpiler: No kernels found in code object\n";
    return Result;
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raising " << KernelNames.size()
                          << " kernel(s) [" << SourceIsa << " -> " << TargetIsa
                          << "]\n");

  auto Text = extractTextSection(CodeObjectData);
  if (!Text.Valid) {
    llvm::errs() << "transpiler: Failed to extract .text section\n";
    return Result;
  }

  DumpDir TmpDir;
  if (!TmpDir.Valid)
    return Result;

  static const char *SDumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (SDumpInput && SDumpInput[0] == '1')
    writeFile(TmpDir.filePath("input.co"), CodeObjectData);

  std::vector<std::string> ObjPaths;
  for (size_t I = 0; I < KernelNames.size(); ++I) {
    const auto &KName = KernelNames[I];
    std::string ObjPath = TmpDir.filePath("k" + std::to_string(I) + ".o");

    LLVM_DEBUG(llvm::dbgs() << "transpiler:   [" << (I + 1) << "/"
                            << KernelNames.size() << "] " << KName << " ... ");

    if (!raiseAndCompileKernel(Text, CodeObjectData, KName,
                               SourceIsa, TargetIsa, TmpDir, ObjPath, Result)) {
      LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
      Result.Success = false;
      return Result;
    }
    LLVM_DEBUG(llvm::dbgs() << "OK\n");
    ObjPaths.push_back(std::move(ObjPath));
  }

  std::string HsacoPath = TmpDir.filePath("merged.hsaco");
  if (!linkObjects(ObjPaths, HsacoPath))
    return Result;

  Result.Hsaco = readFile(HsacoPath);
  if (Result.Hsaco.empty()) {
    llvm::errs() << "transpiler: Failed to read merged HSACO\n";
    return Result;
  }
  collectTargetPrivateSegmentMetadata(Result, KernelNames);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Merged HSACO: " << Result.Hsaco.size()
                          << " bytes, " << KernelNames.size()
                          << " kernel(s)\n");
  Result.Success = true;
  return Result;
}

} // namespace COMGR::hotswap
