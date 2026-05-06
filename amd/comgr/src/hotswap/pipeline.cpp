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
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
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

// Read a file into a vector via llvm::MemoryBuffer; returns empty on failure.
std::vector<uint8_t> readFile(llvm::StringRef path) {
  auto BufOrErr = llvm::MemoryBuffer::getFile(path, /*IsText=*/false);
  if (!BufOrErr) {
    llvm::errs() << "transpiler: Cannot read file: " << path << ": "
                 << BufOrErr.getError().message() << "\n";
    return {};
  }
  llvm::StringRef Data = (*BufOrErr)->getBuffer();
  return std::vector<uint8_t>(Data.bytes_begin(), Data.bytes_end());
}

bool writeFile(llvm::StringRef path, llvm::StringRef contents) {
  std::ofstream f(path.str());
  if (!f.is_open()) {
    llvm::errs() << "transpiler: Cannot write file: " << path << "\n";
    return false;
  }
  f.write(contents.data(), contents.size());
  f.flush();
  if (!f) {
    llvm::errs() << "transpiler: write failed for: " << path << "\n";
    return false;
  }
  return true;
}

bool writeFile(llvm::StringRef path, llvm::ArrayRef<uint8_t> data) {
  std::ofstream f(path.str(), std::ios::binary);
  if (!f.is_open()) {
    llvm::errs() << "transpiler: Cannot write file: " << path << "\n";
    return false;
  }
  f.write(reinterpret_cast<const char *>(data.data()), data.size());
  f.flush();
  if (!f) {
    llvm::errs() << "transpiler: write failed for: " << path << "\n";
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
std::string makeSafeBasename(llvm::StringRef kernelName,
                             size_t reservedSuffixBytes = 8) {
  constexpr size_t kMaxComponentBytes = 255;
  if (kernelName.size() + reservedSuffixBytes <= kMaxComponentBytes)
    return kernelName.str();

  // FNV-1a 64-bit hash — small, deterministic, no libstdc++ dep beyond cstdint.
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : kernelName) {
    h ^= c;
    h *= 0x100000001b3ull;
  }

  constexpr size_t kHashHexBytes = 16;   // "%016llx"
  constexpr size_t kSeparatorBytes = 1;  // '_'
  const size_t prefixBudget = kMaxComponentBytes - reservedSuffixBytes -
                              kHashHexBytes - kSeparatorBytes;
  std::string prefix = kernelName.substr(0, prefixBudget).str();
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx",
                static_cast<unsigned long long>(h));
  return prefix + "_" + buf;
}

int toolTimeoutSeconds() {
  static const int timeout = [] {
    constexpr int kDefaultTimeoutSeconds = 300;
    const char *env = std::getenv("HSA_HOTSWAP_TOOL_TIMEOUT_S");
    if (!env || !env[0])
      return kDefaultTimeoutSeconds;
    char *end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (*end != '\0' || parsed <= 0) {
      llvm::errs() << "transpiler: invalid HSA_HOTSWAP_TOOL_TIMEOUT_S='"
                   << env << "'; using default " << kDefaultTimeoutSeconds
                   << " seconds\n";
      return kDefaultTimeoutSeconds;
    }
    return static_cast<int>(parsed);
  }();
  return timeout;
}

int runTool(llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args) {
  LLVM_DEBUG({
    llvm::dbgs() << "transpiler: Running:";
    for (auto &a : args) llvm::dbgs() << " " << a;
    llvm::dbgs() << "\n";
  });

  auto exeOrErr = llvm::sys::findProgramByName(program);
  if (!exeOrErr) {
    llvm::errs() << "transpiler: tool not found: " << program << "\n";
    return -1;
  }

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(*exeOrErr, args, /*Env=*/std::nullopt,
                                     /*Redirects=*/{},
                                     /*SecondsToWait=*/toolTimeoutSeconds(),
                                     /*MemoryLimit=*/0, &errMsg);
  if (rc != 0)
    llvm::errs() << "transpiler: " << program << " failed (exit " << rc << ")"
                 << (errMsg.empty() ? "" : ": " + errMsg) << "\n";
  return rc;
}

struct DumpDir {
  llvm::SmallString<128> path;
  bool valid = false;
  bool persistent = false;

  DumpDir() {
    static const char *envDir = std::getenv("HSA_HOTSWAP_DUMP_DIR");
    if (envDir && envDir[0]) {
      persistent = true;
      path = envDir;
      if (auto ec = llvm::sys::fs::create_directories(path)) {
        llvm::errs() << "hotswap: failed to create dump dir '"
                     << path << "': " << ec.message() << "\n";
        return;
      }
      // Create a unique subdirectory per invocation so parallel runs
      // don't clobber each other.
      llvm::SmallString<128> sub;
      if (auto ec = llvm::sys::fs::createUniqueDirectory(
              path + "/hotswap", sub)) {
        llvm::errs() << "hotswap: failed to create subdir in '"
                     << path << "': " << ec.message() << "\n";
        return;
      }
      path = sub;
      valid = true;
    } else {
      if (auto ec =
              llvm::sys::fs::createUniqueDirectory("transpiler", path)) {
        llvm::errs() << "hotswap: failed to create temp dir: "
                     << ec.message() << "\n";
      } else {
        valid = true;
      }
    }
  }

  ~DumpDir() {
    if (valid && !persistent)
      llvm::sys::fs::remove_directories(path);
  }

  DumpDir(const DumpDir &) = delete;
  DumpDir &operator=(const DumpDir &) = delete;

  std::string filePath(llvm::StringRef name) const {
    llvm::SmallString<256> p(path);
    llvm::sys::path::append(p, name);
    return std::string(p);
  }
};

} // anonymous namespace

static thread_local bool StrictModeOverrideActive = false;
static thread_local bool StrictModeOverrideValue = false;

ScopedStrictMode::ScopedStrictMode(bool Enabled)
    : PreviousActive(StrictModeOverrideActive),
      PreviousValue(StrictModeOverrideValue) {
  StrictModeOverrideActive = true;
  StrictModeOverrideValue = Enabled;
}

ScopedStrictMode::~ScopedStrictMode() {
  StrictModeOverrideActive = PreviousActive;
  StrictModeOverrideValue = PreviousValue;
}

bool isStrictMode() {
  if (StrictModeOverrideActive)
    return StrictModeOverrideValue;

  // Parsed once on first call. The handler implementations call this on
  // every relevant instruction, so going through the OS allocator
  // (`std::getenv`) repeatedly would be wasteful; the result also cannot
  // change inside a process because the env var is read once at the
  // first transpile and reused for the rest of the process lifetime.
  // Treats any non-empty value as enabled to keep the runner side
  // (`HSA_HOTSWAP_STRICT=1`) and the pipeline side decoupled; a future
  // shell that writes `HSA_HOTSWAP_STRICT=true` still works.
  static const bool s_strict = []() {
    const char *v = std::getenv("HSA_HOTSWAP_STRICT");
    return v && v[0] != '\0';
  }();
  return s_strict;
}

// Raise one kernel to IR, compile to a relocatable .o via llc + llvm-mc.
// On success, writes the .o to objPath and returns true.
static bool raiseAndCompileKernel(const TextSection &text,
                                  llvm::ArrayRef<uint8_t> codeObjectData,
                                  llvm::StringRef kernelName,
                                  llvm::StringRef sourceISA,
                                  llvm::StringRef targetISA,
                                  const DumpDir &tmpDir,
                                  llvm::StringRef objPath,
                                  PipelineResult &result,
                                  bool enableWritelaneRewrite,
                                  bool enableWaveNative) {
  auto meta = extractKernelMeta(codeObjectData, kernelName);
  if (meta.Args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << kernelName
                 << "', using empty metadata\n";
  }

  auto kernelOffsetOrErr = findKernelSymbolOffset(codeObjectData, kernelName);
  if (!kernelOffsetOrErr) {
    std::string err = llvm::toString(kernelOffsetOrErr.takeError());
    llvm::errs() << "transpiler: " << err << "\n";
    result.FailKernel = kernelName;
    result.FailMnemonic = "__kernel_offset__";
    result.FailReason = "KernelSymbolOffsetLookupFailed";
    result.FailFormat = "KernelSymbolOffsetLookupFailed";
    result.FailDetail = err;
    return false;
  }
  uint64_t kernelOffset = *kernelOffsetOrErr;
  LLVM_DEBUG(if (kernelOffset > 0)
    llvm::dbgs() << "transpiler: Kernel '" << kernelName
                 << "' at .text offset 0x" << llvm::utohexstr(kernelOffset)
                 << "\n");

  auto raised = raiseToIR(text.Bytes, sourceISA, kernelName, meta, kernelOffset,
                           targetISA, enableWritelaneRewrite,
                           enableWaveNative);
  if (!raised.Success) {
    llvm::errs() << "transpiler: Raising '" << kernelName << "' to LLVM IR failed";
    result.FailKernel = kernelName;
    if (!raised.Failure.Mnemonic.empty()) {
      llvm::errs() << " (unsupported: " << raised.Failure.Mnemonic << ")";
      result.FailMnemonic = raised.Failure.Mnemonic;
    }
    if (raised.Failure.hasFailed()) {
      result.FailReason = reasonString(raised.Failure.Reason);
      result.FailFormat = raised.Failure.Format;
      result.FailDetail = raised.Failure.Detail;
      result.FailOffset = raised.Failure.Offset;
    }
    llvm::errs() << "\n";
    return false;
  }
  result.LiftedCount += raised.LiftedCount;
  result.TotalCount += raised.TotalCount;
  if (raised.UsesScratchPrivateSegment) {
    result.UsesScratchPrivateSegment = true;
    if (raised.SourcePrivateSegmentFixedSize >
        result.SourcePrivateSegmentFixedSize)
      result.SourcePrivateSegmentFixedSize = raised.SourcePrivateSegmentFixedSize;
  }
  result.C5SuppressedCount += raised.C5SuppressedCount;
  if (result.C5SuppressionReason.empty() &&
      !raised.C5SuppressionReason.empty())
    result.C5SuppressionReason = raised.C5SuppressionReason;
  if (!result.IrText.empty())
    result.IrText += "\n";
  result.IrText += raised.IrText;

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raised '" << kernelName << "' "
                           << raised.LiftedCount << "/"
                           << raised.TotalCount << " instructions\n");

  // Kernel names from Tensile et al. routinely exceed 255 bytes, which is
  // the per-component limit on ext4/xfs/tmpfs.  makeSafeBasename() hashes
  // the tail and truncates the head when the full name would blow the
  // budget; the symbol name inside the IR stays untouched, so debug
  // tooling can still resolve the long name from the LLVM module.
  std::string fileStem = makeSafeBasename(kernelName, /*reservedSuffixBytes=*/5);
  std::string irPath  = tmpDir.filePath(fileStem + ".ll");
  std::string asmPath = tmpDir.filePath(fileStem + ".s");

  if (!writeFile(irPath, raised.IrText))
    return false;

  static const char *s_dumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (s_dumpInput && s_dumpInput[0] == '1' && !raised.DisasmText.empty())
    writeFile(tmpDir.filePath(fileStem + ".dis"), raised.DisasmText);

  std::string llcBin = std::string(LLVM_TOOLS_DIR) + "/llc";
  std::string mcpuLlc = ("-mcpu=" + targetISA).str();
  if (runTool(llcBin, {llcBin, "-march=amdgcn", mcpuLlc, "-filetype=asm", "-o",
                       asmPath, irPath}) != 0) {
    llvm::errs() << "transpiler: llc failed for '" << kernelName << "'\n";
    return false;
  }

  {
    auto asmData = readFile(asmPath);
    if (!result.AsmText.empty())
      result.AsmText += "\n";
    result.AsmText.append(asmData.begin(), asmData.end());
  }

  std::string mcBin = std::string(LLVM_TOOLS_DIR) + "/llvm-mc";
  std::string mcpuMc = ("-mcpu=" + targetISA).str();
  if (runTool(mcBin, {mcBin, "-triple=amdgcn-amd-amdhsa", mcpuMc,
                      "-filetype=obj", "-o", objPath, asmPath}) != 0) {
    llvm::errs() << "transpiler: llvm-mc failed for '" << kernelName << "'\n";
    return false;
  }

  return true;
}

// Link one or more relocatable .o files into a shared HSACO.
static bool linkObjects(llvm::ArrayRef<std::string> objPaths,
                        llvm::StringRef hsacoPath) {
  std::string lldBin = std::string(LLVM_TOOLS_DIR) + "/ld.lld";
  llvm::SmallVector<llvm::StringRef, 16> args;
  args.push_back(lldBin);
  args.push_back("-shared");
  args.push_back("-o");
  args.push_back(hsacoPath);
  for (auto &o : objPaths)
    args.push_back(o);
  if (runTool(lldBin, args) != 0) {
    llvm::errs() << "transpiler: ld.lld failed\n";
    return false;
  }
  return true;
}

void collectTargetPrivateSegmentMetadata(PipelineResult &result,
                                         llvm::ArrayRef<std::string> kernelNames) {
  using namespace llvm::amdhsa;
  if (result.Hsaco.empty())
    return;
  for (llvm::StringRef kernelName : kernelNames) {
    KernelMeta meta = extractKernelMeta(result.Hsaco, kernelName);
    if (!meta.HasKernelDescriptor)
      continue;
    result.TargetPrivateSegmentFixedSize = std::max(
        result.TargetPrivateSegmentFixedSize,
        meta.PrivateSegmentFixedSize);
    const bool enabled =
        (meta.ComputePgmRsrc2 &
         (1u << COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT)) != 0;
    result.TargetEnablePrivateSegment |= enabled;
  }
}

PipelineResult runPipeline(llvm::ArrayRef<uint8_t> codeObjectData,
                           llvm::StringRef sourceISA,
                           llvm::StringRef targetISA,
                           llvm::StringRef kernelName,
                           bool enableWritelaneRewrite,
                           bool enableWaveNative) {
  PipelineResult result;

  auto text = extractTextSection(codeObjectData);
  if (!text.Valid) {
    llvm::errs() << "transpiler: Failed to extract .text section\n";
    return result;
  }

  DumpDir tmpDir;
  if (!tmpDir.valid)
    return result;

  {
    static const char *s_dumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (s_dumpInput && s_dumpInput[0] == '1')
      writeFile(tmpDir.filePath("input.co"), codeObjectData);
  }

  std::string objPath   = tmpDir.filePath("kernel.o");
  std::string hsacoPath = tmpDir.filePath("kernel.hsaco");

  if (!raiseAndCompileKernel(text, codeObjectData, kernelName,
                             sourceISA, targetISA, tmpDir, objPath, result,
                             enableWritelaneRewrite, enableWaveNative))
    return result;

  if (!linkObjects({objPath}, hsacoPath))
    return result;

  result.Hsaco = readFile(hsacoPath);
  if (result.Hsaco.empty()) {
    llvm::errs() << "transpiler: Failed to read HSACO\n";
    return result;
  }
  std::string kernelNameStr = kernelName.str();
  collectTargetPrivateSegmentMetadata(result, {kernelNameStr});

  LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: " << result.Hsaco.size()
                          << " bytes\n");
  result.Success = true;
  return result;
}

PipelineResult runPipelineAllKernels(llvm::ArrayRef<uint8_t> codeObjectData,
                                     llvm::StringRef sourceISA,
                                     llvm::StringRef targetISA,
                                     bool enableWritelaneRewrite,
                                     bool enableWaveNative) {
  PipelineResult result;

  auto kernelNames = listKernelNames(codeObjectData);
  if (kernelNames.empty()) {
    llvm::errs() << "transpiler: No kernels found in code object\n";
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raising " << kernelNames.size()
                          << " kernel(s) [" << sourceISA << " -> " << targetISA
                          << "]\n");

  auto text = extractTextSection(codeObjectData);
  if (!text.Valid) {
    llvm::errs() << "transpiler: Failed to extract .text section\n";
    return result;
  }

  DumpDir tmpDir;
  if (!tmpDir.valid)
    return result;

  static const char *s_dumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (s_dumpInput && s_dumpInput[0] == '1')
    writeFile(tmpDir.filePath("input.co"), codeObjectData);

  std::vector<std::string> objPaths;
  for (size_t i = 0; i < kernelNames.size(); ++i) {
    const auto &kName = kernelNames[i];
    std::string objPath = tmpDir.filePath("k" + std::to_string(i) + ".o");

    LLVM_DEBUG(llvm::dbgs() << "transpiler:   [" << (i + 1) << "/"
                            << kernelNames.size() << "] " << kName << " ... ");

    if (!raiseAndCompileKernel(text, codeObjectData, kName,
                               sourceISA, targetISA, tmpDir, objPath, result,
                               enableWritelaneRewrite, enableWaveNative)) {
      LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
      result.Success = false;
      return result;
    }
    LLVM_DEBUG(llvm::dbgs() << "OK\n");
    objPaths.push_back(std::move(objPath));
  }

  std::string hsacoPath = tmpDir.filePath("merged.hsaco");
  if (!linkObjects(objPaths, hsacoPath))
    return result;

  result.Hsaco = readFile(hsacoPath);
  if (result.Hsaco.empty()) {
    llvm::errs() << "transpiler: Failed to read merged HSACO\n";
    return result;
  }
  collectTargetPrivateSegmentMetadata(result, kernelNames);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Merged HSACO: " << result.Hsaco.size()
                          << " bytes, " << kernelNames.size()
                          << " kernel(s)\n");
  result.Success = true;
  return result;
}

} // namespace COMGR::hotswap
