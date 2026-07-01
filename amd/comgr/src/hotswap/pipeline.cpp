#include "pipeline.h"
#include "code-object-utils.h"
#include "mc-state.h"
#include "raise-failure.h"
#include "raiser.h"

#include "lld/Common/CommonLinkerContext.h"
#include "lld/Common/Driver.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <string>

LLD_HAS_DRIVER(elf)

#define DEBUG_TYPE "transpiler"

namespace COMGR::hotswap {

namespace {

using TimingClock = std::chrono::steady_clock;

double secondsBetween(TimingClock::time_point Start,
                      TimingClock::time_point End) {
  return std::chrono::duration<double>(End - Start).count();
}

TimingClock::time_point timingStart(bool CollectTimings) {
  return CollectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool CollectTimings, TimingClock::time_point Start) {
  return CollectTimings ? secondsBetween(Start, TimingClock::now()) : 0.0;
}

llvm::Error writeFile(llvm::StringRef Path, llvm::StringRef Bytes,
                      llvm::sys::fs::OpenFlags Flags) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, Flags);

  if (EC)
    return llvm::createFileError(Path, EC);

  Out.write(Bytes.data(), Bytes.size());
  Out.flush();

  if (Out.has_error())
    return llvm::createFileError(Path, Out.error());

  return llvm::Error::success();
}

llvm::Error writeFile(llvm::StringRef Path, llvm::StringRef Contents) {
  return writeFile(Path, Contents, llvm::sys::fs::OF_Text);
}

llvm::Error writeFile(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data) {
  return writeFile(
      Path,
      llvm::StringRef(reinterpret_cast<const char *>(Data.data()), Data.size()),
      llvm::sys::fs::OF_None);
}

// Best-effort write of a debug artifact: log and swallow any failure so a dump
// error never aborts the raise/compile pipeline.
void writeDebugFile(llvm::StringRef Path, llvm::StringRef Contents) {
  llvm::logAllUnhandledErrors(writeFile(Path, Contents), llvm::errs(),
                              "transpiler: ");
}

void writeDebugFile(llvm::StringRef Path, llvm::ArrayRef<uint8_t> Data) {
  llvm::logAllUnhandledErrors(writeFile(Path, Data), llvm::errs(),
                              "transpiler: ");
}

// Derive a filesystem-safe basename for an arbitrarily long kernel name.
// Most POSIX filesystems cap individual path components at 255 bytes, and
// Hotswap generates sibling files off the same stem (e.g. `<stem>.ll`,
// `<stem>.s`, `<stem>.dis`), so we leave a small suffix budget and fold
// anything longer down to a deterministic truncated+hashed form so two
// kernels with a shared 240-byte prefix don't collide on disk.
//
// The returned basename preserves a readable prefix of the original name
// for debuggability; it's only intended for temp-dir scratch files --
// symbol names inside the IR itself are unaffected.
std::string makeSafeBasename(llvm::StringRef KernelName,
                             size_t ReservedSuffixBytes = 8) {
  constexpr size_t MaxComponentBytes = 255;
  if (KernelName.size() + ReservedSuffixBytes <= MaxComponentBytes)
    return KernelName.str();

  uint64_t H = llvm::xxh3_64bits(KernelName);

  constexpr size_t HashHexBytes = 16;  // 64-bit hash as hex
  constexpr size_t SeparatorBytes = 1; // '_'
  const size_t PrefixBudget =
      MaxComponentBytes - ReservedSuffixBytes - HashHexBytes - SeparatorBytes;
  std::string Prefix = KernelName.substr(0, PrefixBudget).str();
  std::string Hex = llvm::utohexstr(H, /*LowerCase=*/true, /*Width=*/16);
  return Prefix + "_" + Hex;
}

llvm::OptimizationLevel toOptimizationLevel(unsigned Level) {
  switch (Level) {
  case 0:
    return llvm::OptimizationLevel::O0;
  case 1:
    return llvm::OptimizationLevel::O1;
  case 2:
    return llvm::OptimizationLevel::O2;
  default:
    return llvm::OptimizationLevel::O3;
  }
}

std::unique_ptr<llvm::TargetMachine>
createHotswapTargetMachine(llvm::StringRef TargetISA, unsigned OptLevel) {
  std::string Err;
  llvm::Triple TheTriple(kAMDGPUTriple);
  const llvm::Target *TheTarget =
      llvm::TargetRegistry::lookupTarget(TheTriple, Err);
  // The triple is hardcoded and the AMDGPU target is linked in, so a lookup
  // miss is a build misconfiguration rather than a recoverable error.
  if (!TheTarget)
    llvm::report_fatal_error(
        llvm::Twine("transpiler: AMDGPU target not registered: ") + Err);
  llvm::CodeGenOptLevel CGOL = llvm::CodeGenOpt::getLevel(OptLevel).value_or(
      llvm::CodeGenOptLevel::Default);
  llvm::TargetOptions Opts;
  return std::unique_ptr<llvm::TargetMachine>(TheTarget->createTargetMachine(
      TheTriple, TargetISA, /*Features=*/"", Opts, llvm::Reloc::PIC_,
      /*CodeModel=*/std::nullopt, CGOL));
}

// In-process `opt`: run the default per-module pipeline at OptLevel.
void runOptPipeline(llvm::Module &M, llvm::TargetMachine &TM,
                    unsigned OptLevel) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB(&TM);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  llvm::OptimizationLevel OL = toOptimizationLevel(OptLevel);
  llvm::ModulePassManager MPM = OL == llvm::OptimizationLevel::O0
                                    ? PB.buildO0DefaultPipeline(OL)
                                    : PB.buildPerModuleDefaultPipeline(OL);
  MPM.run(M, MAM);
}

// In-process `llc`: run codegen for `M` and emit `FileType` to `OS`.
llvm::Error emitCodeGen(llvm::Module &M, llvm::TargetMachine &TM,
                        llvm::CodeGenFileType FileType,
                        llvm::raw_pwrite_stream &OS) {
  llvm::legacy::PassManager PM;
  if (TM.addPassesToEmitFile(PM, OS, /*DwoOut=*/nullptr, FileType))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "target cannot emit requested file type");

  PM.run(M);
  return llvm::Error::success();
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
      if (auto EC = llvm::sys::fs::create_directories(Path)) {
        llvm::errs() << "hotswap: failed to create dump dir '" << Path
                     << "': " << EC.message() << "\n";
        return;
      }
      // Create a unique subdirectory per invocation so parallel runs
      // don't clobber each other.
      llvm::SmallString<128> Sub;
      if (auto EC =
              llvm::sys::fs::createUniqueDirectory(Path + "/hotswap", Sub)) {
        llvm::errs() << "hotswap: failed to create subdir in '" << Path
                     << "': " << EC.message() << "\n";
        return;
      }
      Path = Sub;
      Valid = true;
    } else {
      if (auto EC = llvm::sys::fs::createUniqueDirectory("transpiler", Path)) {
        llvm::errs() << "hotswap: failed to create temp dir: " << EC.message()
                     << "\n";
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
  static const bool Strict = []() {
    const char *V = std::getenv("HSA_HOTSWAP_STRICT");
    return V && V[0] != '\0';
  }();
  return Strict;
}

// Raise one kernel to IR, then opt + codegen it to a relocatable .o.
// On success, writes the .o to ObjPath and returns the kernel's results.
// When `Stats` is non-null, per-kernel counts and (if enabled) timing samples
// are accumulated into it.
static llvm::Expected<PipelineResult> raiseAndCompileKernel(
    const TextSection &Text, llvm::MemoryBufferRef CodeObjectData,
    llvm::StringRef KernelName, llvm::StringRef SourceISA,
    llvm::StringRef TargetISA, const DumpDir &TmpDir, llvm::StringRef ObjPath,
    PipelineStats *Stats, const PipelineOptions &Options) {
  const bool CollectTimings = Options.CollectTimings;
  PipelineResult Result;

  auto RaiseStart = timingStart(CollectTimings);
  llvm::Expected<KernelMeta> MetaOrErr =
      extractKernelMeta(CodeObjectData, KernelName);
  if (!MetaOrErr) {
    llvm::logAllUnhandledErrors(MetaOrErr.takeError(), llvm::errs());
  }

  KernelMeta Meta = MetaOrErr ? std::move(*MetaOrErr) : KernelMeta{};
  if (Meta.Args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << KernelName
                 << "', using empty metadata\n";
  }

  auto KernelExtentOrErr = findKernelSymbolExtent(CodeObjectData, KernelName);
  if (!KernelExtentOrErr) {
    if (Stats)
      Stats->Timings.raiseSeconds += timingElapsed(CollectTimings, RaiseStart);

    return KernelExtentOrErr.takeError();
  }
  uint64_t KernelOffset = KernelExtentOrErr->Offset;
  uint64_t KernelSize = KernelExtentOrErr->Size;
  LLVM_DEBUG(if (KernelOffset > 0) llvm::dbgs()
             << "transpiler: Kernel '" << KernelName << "' at .text offset 0x"
             << llvm::utohexstr(KernelOffset) << " size 0x"
             << llvm::utohexstr(KernelSize) << "\n");

  RaiseStats KernelStats;
  llvm::Expected<RaiseResult> RaisedOrErr = raiseToIR(
      Text.Bytes, SourceISA, KernelName, Meta, KernelOffset, KernelSize,
      TargetISA, Options.EnableWritelaneRewrite, Options.EnableWaveNative,
      Options.AssumeHipGlobalOffsetZero, &KernelStats);
  if (!RaisedOrErr) {
    if (Stats)
      Stats->Timings.raiseSeconds += timingElapsed(CollectTimings, RaiseStart);
    return RaisedOrErr.takeError();
  }
  RaiseResult Raised = std::move(*RaisedOrErr);
  if (Stats) {
    Stats->LiftedCount += KernelStats.LiftedCount;
    Stats->TotalCount += KernelStats.TotalCount;
    if (KernelStats.UsesScratchPrivateSegment) {
      Stats->UsesScratchPrivateSegment = true;
      Stats->SourcePrivateSegmentFixedSize =
          std::max(Stats->SourcePrivateSegmentFixedSize,
                   KernelStats.SourcePrivateSegmentFixedSize);
    }
    Stats->C5SuppressedCount += KernelStats.C5SuppressedCount;
    if (Stats->C5SuppressionReason.empty() &&
        !KernelStats.C5SuppressionReason.empty())
      Stats->C5SuppressionReason = std::move(KernelStats.C5SuppressionReason);
    Stats->Timings.raiseSeconds += timingElapsed(CollectTimings, RaiseStart);
  }
  Result.IrText = Raised.IrText;

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raised '" << KernelName << "' "
                          << KernelStats.LiftedCount << "/"
                          << KernelStats.TotalCount << " instructions\n");

  // Kernel names from Tensile et al. routinely exceed 255 bytes, which is
  // the per-component limit on ext4/xfs/tmpfs.  makeSafeBasename() hashes
  // the tail and truncates the head when the full name would blow the
  // budget; the symbol name inside the IR stays untouched, so debug
  // tooling can still resolve the long name from the LLVM module.
  std::string FileStem =
      makeSafeBasename(KernelName, /*ReservedSuffixBytes=*/5);

  // Codegen consumes the in-memory module directly; the .ll/.s/.dis files are
  // debug dumps only, so skip them unless a persistent dump dir was set (a
  // non-persistent temp dir is deleted on exit, taking the dumps with it).
  auto WriteIrStart = timingStart(CollectTimings);
  if (TmpDir.Persistent) {
    writeDebugFile(TmpDir.filePath(FileStem + ".ll"), Raised.IrText);
    static const char *DumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (DumpInput && DumpInput[0] == '1' && !Raised.DisasmText.empty())
      writeDebugFile(TmpDir.filePath(FileStem + ".dis"), Raised.DisasmText);
  }
  if (Stats)
    Stats->Timings.writeIrSeconds +=
        timingElapsed(CollectTimings, WriteIrStart);

  if (!Raised.Module)
    return llvm::createStringError("raiser produced no module for '" +
                                   KernelName + "'");
  llvm::Module &M = *Raised.Module;

  std::unique_ptr<llvm::TargetMachine> TM =
      createHotswapTargetMachine(TargetISA, Options.OptLevel);
  if (!TM)
    return llvm::createStringError("failed to create TargetMachine for '" +
                                   KernelName + "'");
  M.setDataLayout(TM->createDataLayout());

  auto OptStart = timingStart(CollectTimings);
  runOptPipeline(M, *TM, Options.OptLevel);
  if (Stats)
    Stats->Timings.optSeconds += timingElapsed(CollectTimings, OptStart);

  // Object codegen consumes the module, so clone it first when a debug
  // assembly dump is still needed.
  std::unique_ptr<llvm::Module> AsmModule;
  if (TmpDir.Persistent)
    AsmModule = llvm::CloneModule(M);

  llvm::SmallVector<char, 4096> ObjBytes;
  auto LlcStart = timingStart(CollectTimings);
  llvm::raw_svector_ostream OS(ObjBytes);
  if (llvm::Error Err =
          emitCodeGen(M, *TM, llvm::CodeGenFileType::ObjectFile, OS))
    return Err;

  if (Stats)
    Stats->Timings.llcSeconds += timingElapsed(CollectTimings, LlcStart);

  if (llvm::Error Err = writeFile(
          ObjPath, llvm::ArrayRef<uint8_t>(
                       reinterpret_cast<const uint8_t *>(ObjBytes.data()),
                       ObjBytes.size())))
    return Err;

  // Textual assembly is a debug-only artifact emitted from the clone so the
  // object codegen above stays the canonical lowering.
  if (AsmModule) {
    llvm::SmallString<4096> AsmText;
    llvm::raw_svector_ostream OS(AsmText);
    if (llvm::Error Err = emitCodeGen(*AsmModule, *TM,
                                      llvm::CodeGenFileType::AssemblyFile, OS))
      llvm::consumeError(
          std::move(Err)); // debug artifact; never fail the build
    else
      writeDebugFile(TmpDir.filePath(FileStem + ".s"), AsmText);
  }

  return Result;
}

// Link one or more relocatable .o files into a shared HSACO using the
// in-process LLD ELF driver.
static llvm::Error linkObjects(llvm::ArrayRef<std::string> ObjPaths,
                               llvm::StringRef HsacoPath) {
  std::string HsacoPathStr = HsacoPath.str();
  llvm::SmallVector<const char *, 16> Args;
  Args.push_back("ld.lld");
  Args.push_back("-shared");
  Args.push_back("--threads=1");
  Args.push_back("-o");
  Args.push_back(HsacoPathStr.c_str());
  for (auto &O : ObjPaths)
    Args.push_back(O.c_str());

  // lld::lldMain drives a process-global CommonLinkerContext and is neither
  // re-entrant nor thread-safe; serialize all in-process links.
  static std::mutex LldMutex;
  std::lock_guard<std::mutex> LldLock(LldMutex);
  std::string OutString;
  std::string ErrString;
  llvm::raw_string_ostream OutStream(OutString);
  llvm::raw_string_ostream ErrStream(ErrString);
  lld::Result Ret =
      lld::lldMain(Args, OutStream, ErrStream, {{lld::Gnu, &lld::elf::link}});
  lld::CommonLinkerContext::destroy();
  if (Ret.retCode != 0 || !Ret.canRunAgain) {
    ErrStream.flush();
    return llvm::createStringError(
        "ld.lld failed return code: " + llvm::Twine(Ret.retCode) +
        " stderr: " + ErrString);
  }

  return llvm::Error::success();
}

void collectTargetPrivateSegmentMetadata(
    PipelineStats &Stats, llvm::MemoryBufferRef HsacoBuf,
    llvm::ArrayRef<std::string> KernelNames) {
  using namespace llvm::amdhsa;
  if (HsacoBuf.getBufferSize() == 0)
    return;
  for (llvm::StringRef KernelName : KernelNames) {
    llvm::Expected<KernelMeta> MetaOrErr =
        extractKernelMeta(HsacoBuf, KernelName);
    if (!MetaOrErr) {
      llvm::consumeError(MetaOrErr.takeError());
      continue;
    }
    KernelMeta &Meta = *MetaOrErr;
    if (!Meta.HasKernelDescriptor)
      continue;
    Stats.TargetPrivateSegmentFixedSize =
        std::max(Stats.TargetPrivateSegmentFixedSize,
                 static_cast<uint32_t>(Meta.PrivateSegmentFixedSize));
    const bool Enabled =
        (Meta.ComputePgmRsrc2 &
         (1u << COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT)) != 0;
    Stats.TargetEnablePrivateSegment |= Enabled;
  }
}

llvm::Expected<PipelineResult>
runPipeline(llvm::MemoryBufferRef CodeObjectData, llvm::StringRef SourceISA,
            llvm::StringRef TargetISA, llvm::StringRef KernelName,
            PipelineOptions Options, PipelineStats *Stats) {
  const bool CollectTimings = Options.CollectTimings;
  auto TotalStart = timingStart(CollectTimings);

  auto Run = [&]() -> llvm::Expected<PipelineResult> {
    auto ExtractTextStart = timingStart(CollectTimings);
    llvm::Expected<TextSection> TextOrErr = extractTextSection(CodeObjectData);
    if (Stats)
      Stats->Timings.extractTextSeconds =
          timingElapsed(CollectTimings, ExtractTextStart);
    if (!TextOrErr)
      return TextOrErr.takeError();

    TextSection Text = std::move(*TextOrErr);

    auto TempDirStart = timingStart(CollectTimings);
    DumpDir TmpDir;
    if (Stats)
      Stats->Timings.createTempDirSeconds =
          timingElapsed(CollectTimings, TempDirStart);
    if (!TmpDir.Valid)
      return llvm::createStringError("failed to create temp dir");

    {
      static const char *DumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
      if (DumpInput && DumpInput[0] == '1')
        writeDebugFile(TmpDir.filePath("input.co"),
                       llvm::ArrayRef(reinterpret_cast<const uint8_t *>(
                                          CodeObjectData.getBufferStart()),
                                      CodeObjectData.getBufferSize()));
    }

    std::string ObjPath = TmpDir.filePath("kernel.o");
    std::string HsacoPath = TmpDir.filePath("kernel.Hsaco");

    llvm::Expected<PipelineResult> ResultOrErr =
        raiseAndCompileKernel(Text, CodeObjectData, KernelName, SourceISA,
                              TargetISA, TmpDir, ObjPath, Stats, Options);
    if (!ResultOrErr)
      return ResultOrErr.takeError();
    PipelineResult Result = std::move(*ResultOrErr);

    auto LinkStart = timingStart(CollectTimings);
    if (llvm::Error Err = linkObjects({ObjPath}, HsacoPath))
      return Err;

    if (Stats)
      Stats->Timings.linkSeconds += timingElapsed(CollectTimings, LinkStart);

    auto ReadHsacoStart = timingStart(CollectTimings);
    auto HsacoBufOrErr =
        llvm::MemoryBuffer::getFile(HsacoPath, /*IsText=*/false);
    if (!HsacoBufOrErr) {
      return llvm::createFileError(HsacoPath, HsacoBufOrErr.getError());
    }
    Result.Hsaco = std::move(*HsacoBufOrErr);

    if (Stats)
      Stats->Timings.readHsacoSeconds +=
          timingElapsed(CollectTimings, ReadHsacoStart);
    if (Result.Hsaco->getBufferSize() == 0)
      return llvm::createStringError("HSACO buffer is empty");

    std::string KernelNameStr = KernelName.str();
    auto CollectMetadataStart = timingStart(CollectTimings);
    if (Stats) {
      collectTargetPrivateSegmentMetadata(
          *Stats, Result.Hsaco->getMemBufferRef(), {KernelNameStr});
      Stats->Timings.collectMetadataSeconds +=
          timingElapsed(CollectTimings, CollectMetadataStart);
    }

    LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: "
                            << Result.Hsaco->getBufferSize() << " bytes\n");
    return Result;
  };

  llvm::Expected<PipelineResult> Out = Run();
  if (Stats)
    Stats->Timings.totalSeconds = timingElapsed(CollectTimings, TotalStart);
  return Out;
}

llvm::Expected<PipelineResult>
runPipelineAllKernels(llvm::MemoryBufferRef CodeObjectData,
                      llvm::StringRef SourceISA, llvm::StringRef TargetISA,
                      PipelineOptions Options, PipelineStats *Stats) {
  const bool CollectTimings = Options.CollectTimings;
  auto TotalStart = timingStart(CollectTimings);

  auto Run = [&]() -> llvm::Expected<PipelineResult> {
    auto ListKernelsStart = timingStart(CollectTimings);
    llvm::Expected<llvm::SmallVector<std::string>> KernelNamesOrErr =
        listKernelNames(CodeObjectData);
    if (Stats)
      Stats->Timings.listKernelsSeconds =
          timingElapsed(CollectTimings, ListKernelsStart);
    if (!KernelNamesOrErr)
      return KernelNamesOrErr.takeError();

    llvm::SmallVector<std::string> KernelNames = std::move(*KernelNamesOrErr);
    if (KernelNames.empty())
      return llvm::createStringError("no kernels found in code object");

    LLVM_DEBUG(llvm::dbgs()
               << "transpiler: Raising " << KernelNames.size() << " kernel(s) ["
               << SourceISA << " -> " << TargetISA << "]\n");

    auto ExtractTextStart = timingStart(CollectTimings);
    llvm::Expected<TextSection> TextOrErr = extractTextSection(CodeObjectData);
    if (Stats)
      Stats->Timings.extractTextSeconds =
          timingElapsed(CollectTimings, ExtractTextStart);
    if (!TextOrErr)
      return TextOrErr.takeError();

    TextSection Text = std::move(*TextOrErr);
    auto TempDirStart = timingStart(CollectTimings);
    DumpDir TmpDir;
    if (Stats)
      Stats->Timings.createTempDirSeconds =
          timingElapsed(CollectTimings, TempDirStart);
    if (!TmpDir.Valid)
      return llvm::createStringError("failed to create temp dir");

    static const char *DumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (DumpInput && DumpInput[0] == '1')
      writeDebugFile(TmpDir.filePath("input.co"),
                     llvm::ArrayRef(reinterpret_cast<const uint8_t *>(
                                        CodeObjectData.getBufferStart()),
                                    CodeObjectData.getBufferSize()));

    PipelineResult Result;
    std::vector<std::string> ObjPaths;
    for (size_t I = 0; I < KernelNames.size(); ++I) {
      const auto &KName = KernelNames[I];
      std::string ObjPath = TmpDir.filePath("k" + std::to_string(I) + ".o");

      LLVM_DEBUG(llvm::dbgs()
                 << "transpiler:   [" << (I + 1) << "/" << KernelNames.size()
                 << "] " << KName << " ... ");

      llvm::Expected<PipelineResult> KernelOrErr =
          raiseAndCompileKernel(Text, CodeObjectData, KName, SourceISA,
                                TargetISA, TmpDir, ObjPath, Stats, Options);
      if (!KernelOrErr) {
        LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
        return KernelOrErr.takeError();
      }
      LLVM_DEBUG(llvm::dbgs() << "OK\n");

      // Counts and segment stats accumulate directly into `Stats` inside
      // raiseAndCompileKernel; only the merged IR text is stitched here.
      PipelineResult Kernel = std::move(*KernelOrErr);
      if (!Kernel.IrText.empty()) {
        if (!Result.IrText.empty())
          Result.IrText += "\n";
        Result.IrText += Kernel.IrText;
      }
      ObjPaths.push_back(std::move(ObjPath));
    }

    std::string HsacoPath = TmpDir.filePath("merged.Hsaco");
    auto LinkStart = timingStart(CollectTimings);
    if (llvm::Error Err = linkObjects(ObjPaths, HsacoPath))
      return Err;
    if (Stats)
      Stats->Timings.linkSeconds += timingElapsed(CollectTimings, LinkStart);

    auto ReadHsacoStart = timingStart(CollectTimings);
    auto HsacoBufOrErr =
        llvm::MemoryBuffer::getFile(HsacoPath, /*IsText=*/false);

    if (!HsacoBufOrErr)
      return llvm::createFileError(HsacoPath, HsacoBufOrErr.getError());
    Result.Hsaco = std::move(*HsacoBufOrErr);

    if (Stats)
      Stats->Timings.readHsacoSeconds +=
          timingElapsed(CollectTimings, ReadHsacoStart);

    if (Result.Hsaco->getBufferSize() == 0)
      return llvm::createStringError("HSACO buffer is empty");

    auto CollectMetadataStart = timingStart(CollectTimings);
    if (Stats) {
      collectTargetPrivateSegmentMetadata(
          *Stats, Result.Hsaco->getMemBufferRef(), KernelNames);
      Stats->Timings.collectMetadataSeconds +=
          timingElapsed(CollectTimings, CollectMetadataStart);
    }

    LLVM_DEBUG(llvm::dbgs()
               << "transpiler: Merged HSACO: " << Result.Hsaco->getBufferSize()
               << " bytes, " << KernelNames.size() << " kernel(s)\n");
    return Result;
  };

  llvm::Expected<PipelineResult> Out = Run();
  if (Stats)
    Stats->Timings.totalSeconds = timingElapsed(CollectTimings, TotalStart);
  return Out;
}

} // namespace COMGR::hotswap
