#include "pipeline.h"
#include "code-object-utils.h"
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
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/xxhash.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

LLD_HAS_DRIVER(elf)

#define DEBUG_TYPE "transpiler"

namespace COMGR::hotswap {

namespace {

using TimingClock = std::chrono::steady_clock;

double secondsBetween(TimingClock::time_point start,
                      TimingClock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

TimingClock::time_point timingStart(bool CollectTimings) {
  return CollectTimings ? TimingClock::now() : TimingClock::time_point{};
}

double timingElapsed(bool CollectTimings, TimingClock::time_point start) {
  return CollectTimings ? secondsBetween(start, TimingClock::now()) : 0.0;
}

bool writeFile(llvm::StringRef path, llvm::StringRef contents) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "transpiler: Cannot write file: " << path << ": "
                 << EC.message() << "\n";
    return false;
  }
  Out.write(contents.data(), contents.size());
  Out.flush();
  if (Out.has_error()) {
    llvm::errs() << "transpiler: write failed for: " << path << "\n";
    return false;
  }
  return true;
}

bool writeFile(llvm::StringRef path, llvm::ArrayRef<uint8_t> data) {
  std::error_code EC;
  llvm::raw_fd_ostream Out(path, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "transpiler: Cannot write file: " << path << ": "
                 << EC.message() << "\n";
    return false;
  }
  Out.write(reinterpret_cast<const char *>(data.data()), data.size());
  Out.flush();
  if (Out.has_error()) {
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
// for debuggability; it's only intended for temp-dir scratch files --
// symbol names inside the IR itself are unaffected.
std::string makeSafeBasename(llvm::StringRef kernelName,
                             size_t reservedSuffixBytes = 8) {
  constexpr size_t kMaxComponentBytes = 255;
  if (kernelName.size() + reservedSuffixBytes <= kMaxComponentBytes)
    return kernelName.str();

  uint64_t h = llvm::xxh3_64bits(kernelName);

  constexpr size_t kHashHexBytes = 16;   // 64-bit hash as hex
  constexpr size_t kSeparatorBytes = 1;  // '_'
  const size_t prefixBudget = kMaxComponentBytes - reservedSuffixBytes -
                              kHashHexBytes - kSeparatorBytes;
  std::string prefix = kernelName.substr(0, prefixBudget).str();
  std::string hex = llvm::utohexstr(h, /*LowerCase=*/true, /*Width=*/16);
  return prefix + "_" + hex;
}

constexpr llvm::StringRef kHotswapTriple = "amdgcn-amd-amdhsa";

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

llvm::CodeGenOptLevel toCodeGenOptLevel(unsigned Level) {
  switch (Level) {
  case 0:
    return llvm::CodeGenOptLevel::None;
  case 1:
    return llvm::CodeGenOptLevel::Less;
  case 2:
    return llvm::CodeGenOptLevel::Default;
  default:
    return llvm::CodeGenOptLevel::Aggressive;
  }
}

std::unique_ptr<llvm::TargetMachine>
createHotswapTargetMachine(llvm::StringRef targetISA, unsigned OptLevel) {
  std::string err;
  llvm::Triple triple(kHotswapTriple);
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target) {
    llvm::errs() << "transpiler: lookupTarget failed: " << err << "\n";
    return nullptr;
  }
  llvm::TargetOptions opts;
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      triple, targetISA, /*Features=*/"", opts, llvm::Reloc::PIC_,
      /*CodeModel=*/std::nullopt, toCodeGenOptLevel(OptLevel)));
}

// In-process `opt`: run the default per-module pipeline at OptLevel.
llvm::Error runOptPipeline(llvm::Module &M, llvm::TargetMachine &TM,
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
  return llvm::Error::success();
}

// In-process `llc`: lower the module to target assembly text.
llvm::Error emitAssembly(llvm::Module &M, llvm::TargetMachine &TM,
                         std::string &asmOut) {
  llvm::SmallString<4096> buffer;
  llvm::raw_svector_ostream OS(buffer);
  llvm::legacy::PassManager PM;
  if (TM.addPassesToEmitFile(PM, OS, /*DwoOut=*/nullptr,
                             llvm::CodeGenFileType::AssemblyFile))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "target cannot emit assembly");
  PM.run(M);
  asmOut.assign(buffer.begin(), buffer.end());
  return llvm::Error::success();
}

// In-process `llvm-mc`: assemble target assembly text into a relocatable
// object, written to objOut.
bool assembleToObject(llvm::StringRef asmText, llvm::StringRef targetISA,
                      llvm::SmallVectorImpl<char> &objOut) {
  std::string err;
  llvm::Triple triple(kHotswapTriple);
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, err);
  if (!target) {
    llvm::errs() << "transpiler: lookupTarget failed: " << err << "\n";
    return false;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBufferCopy(asmText, "<hotswap-asm>"),
      llvm::SMLoc());

  std::unique_ptr<llvm::MCRegisterInfo> MRI(target->createMCRegInfo(triple));
  llvm::MCTargetOptions MCOptions;
  std::unique_ptr<llvm::MCAsmInfo> MAI(
      target->createMCAsmInfo(*MRI, triple, MCOptions));
  std::unique_ptr<llvm::MCSubtargetInfo> STI(
      target->createMCSubtargetInfo(triple, targetISA, /*Features=*/""));
  std::unique_ptr<llvm::MCInstrInfo> MCII(target->createMCInstrInfo());
  if (!MRI || !MAI || !STI || !MCII) {
    llvm::errs() << "transpiler: failed to create MC info for assembly\n";
    return false;
  }

  llvm::MCContext Ctx(triple, *MAI, *MRI, *STI, &srcMgr);
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI(
      new llvm::MCObjectFileInfo());
  MOFI->initMCObjectFileInfo(Ctx, /*PIC=*/true);
  Ctx.setObjectFileInfo(MOFI.get());

  llvm::raw_svector_ostream Out(objOut);
  llvm::MCCodeEmitter *CE = target->createMCCodeEmitter(*MCII, Ctx);
  llvm::MCAsmBackend *MAB = target->createMCAsmBackend(*STI, *MRI, MCOptions);
  if (!CE || !MAB) {
    llvm::errs() << "transpiler: failed to create MC code emitter/backend\n";
    return false;
  }
  std::unique_ptr<llvm::MCStreamer> Str(target->createMCObjectStreamer(
      triple, Ctx, std::unique_ptr<llvm::MCAsmBackend>(MAB),
      MAB->createObjectWriter(Out), std::unique_ptr<llvm::MCCodeEmitter>(CE),
      *STI));
  Str->initSections(*STI);

  std::unique_ptr<llvm::MCAsmParser> Parser(
      llvm::createMCAsmParser(srcMgr, Ctx, *Str, *MAI));
  std::unique_ptr<llvm::MCTargetAsmParser> TAP(
      target->createMCAsmParser(*STI, *Parser, *MCII));
  if (!TAP) {
    llvm::errs() << "transpiler: failed to create target asm parser\n";
    return false;
  }
  Parser->setTargetParser(*TAP);
  if (Parser->Run(/*NoInitialTextSection=*/false)) {
    llvm::errs() << "transpiler: assembly parsing failed\n";
    return false;
  }
  return true;
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

ScopedStrictMode::ScopedStrictMode(bool enabled)
    : PreviousActive(StrictModeOverrideActive),
      PreviousValue(StrictModeOverrideValue) {
  StrictModeOverrideActive = true;
  StrictModeOverrideValue = enabled;
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
                                  llvm::MemoryBufferRef codeObjectData,
                                  llvm::StringRef kernelName,
                                  llvm::StringRef sourceISA,
                                  llvm::StringRef targetISA,
                                  const DumpDir &tmpDir,
                                  llvm::StringRef objPath,
                                  PipelineResult &result,
                                  const PipelineOptions &options) {
  auto raiseStart = timingStart(options.CollectTimings);
  llvm::Expected<KernelMeta> metaOrErr =
      extractKernelMeta(codeObjectData, kernelName);
  if (!metaOrErr) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << kernelName
                 << "': " << llvm::toString(metaOrErr.takeError())
                 << ", using empty metadata\n";
  }
  KernelMeta meta = metaOrErr ? std::move(*metaOrErr) : KernelMeta{};
  if (meta.Args.empty()) {
    llvm::errs() << "transpiler: WARNING: No metadata found for '" << kernelName
                 << "', using empty metadata\n";
  }

  auto kernelExtentOrErr = findKernelSymbolExtent(codeObjectData, kernelName);
  if (!kernelExtentOrErr) {
    std::string err = llvm::toString(kernelExtentOrErr.takeError());
    llvm::errs() << "transpiler: " << err << "\n";
    result.FailKernel = kernelName;
    result.FailMnemonic = "__kernel_extent__";
    result.FailReason = "KernelSymbolExtentLookupFailed";
    result.FailFormat = "KernelSymbolExtentLookupFailed";
    result.FailDetail = err;
    result.Timings.raiseSeconds += timingElapsed(options.CollectTimings, raiseStart);
    return false;
  }
  uint64_t kernelOffset = kernelExtentOrErr->Offset;
  uint64_t kernelSize = kernelExtentOrErr->Size;
  LLVM_DEBUG(if (kernelOffset > 0)
    llvm::dbgs() << "transpiler: Kernel '" << kernelName
                 << "' at .text offset 0x" << llvm::utohexstr(kernelOffset)
                 << " size 0x" << llvm::utohexstr(kernelSize) << "\n");

  auto raised = raiseToIR(text.Bytes, sourceISA, kernelName, meta, kernelOffset,
                           kernelSize, targetISA,
                           options.EnableWritelaneRewrite,
                           options.EnableWaveNative,
                           options.AssumeHipGlobalOffsetZero);
  if (!raised.Success) {
    llvm::errs() << "transpiler: Raising '" << kernelName << "' to LLVM IR failed";
    result.FailKernel = kernelName;
    RaiseFailure Failure =
        raised.Failure.hasFailed()
            ? raised.Failure
            : RaiseFailure::internalFailure(
                  "raiseToIR returned failure without a structured reason");
    std::string RenderedFailure = formatRaiseFailure(Failure);
    llvm::errs() << " (" << RenderedFailure << ")";
    result.FailMnemonic = Failure.Mnemonic;
    result.FailReason = reasonString(Failure.Reason);
    result.FailFormat = Failure.Format;
    result.FailDetail = RenderedFailure;
    result.FailOffset = Failure.Offset;
    llvm::errs() << "\n";
    result.Timings.raiseSeconds += timingElapsed(options.CollectTimings, raiseStart);
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
  result.Timings.raiseSeconds += timingElapsed(options.CollectTimings, raiseStart);
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

  auto writeIrStart = timingStart(options.CollectTimings);
  if (!writeFile(irPath, raised.IrText))
    return false;

  static const char *s_dumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (s_dumpInput && s_dumpInput[0] == '1' && !raised.DisasmText.empty())
    writeFile(tmpDir.filePath(fileStem + ".dis"), raised.DisasmText);
  result.Timings.writeIrSeconds +=
      timingElapsed(options.CollectTimings, writeIrStart);

  if (!raised.Module) {
    llvm::errs() << "transpiler: raiser produced no module for '" << kernelName
                 << "'\n";
    return false;
  }
  llvm::Module &M = *raised.Module;

  std::unique_ptr<llvm::TargetMachine> TM =
      createHotswapTargetMachine(targetISA, options.OptLevel);
  if (!TM) {
    llvm::errs() << "transpiler: failed to create TargetMachine for '"
                 << kernelName << "'\n";
    return false;
  }
  M.setDataLayout(TM->createDataLayout());

  auto optStart = timingStart(options.CollectTimings);
  if (llvm::Error err = runOptPipeline(M, *TM, options.OptLevel)) {
    result.Timings.optSeconds += timingElapsed(options.CollectTimings, optStart);
    llvm::errs() << "transpiler: opt failed for '" << kernelName
                 << "': " << llvm::toString(std::move(err)) << "\n";
    return false;
  }
  result.Timings.optSeconds += timingElapsed(options.CollectTimings, optStart);

  std::string asmText;
  auto llcStart = timingStart(options.CollectTimings);
  if (llvm::Error err = emitAssembly(M, *TM, asmText)) {
    result.Timings.llcSeconds += timingElapsed(options.CollectTimings, llcStart);
    llvm::errs() << "transpiler: llc failed for '" << kernelName
                 << "': " << llvm::toString(std::move(err)) << "\n";
    return false;
  }
  result.Timings.llcSeconds += timingElapsed(options.CollectTimings, llcStart);

  {
    auto readAsmStart = timingStart(options.CollectTimings);
    if (!result.AsmText.empty())
      result.AsmText += "\n";
    result.AsmText += asmText;
    writeFile(asmPath, asmText);
    result.Timings.readAsmSeconds +=
        timingElapsed(options.CollectTimings, readAsmStart);
  }

  llvm::SmallVector<char, 4096> objBytes;
  auto llvmMcStart = timingStart(options.CollectTimings);
  if (!assembleToObject(asmText, targetISA, objBytes)) {
    result.Timings.llvmMcSeconds +=
        timingElapsed(options.CollectTimings, llvmMcStart);
    llvm::errs() << "transpiler: llvm-mc failed for '" << kernelName << "'\n";
    return false;
  }
  if (!writeFile(objPath, llvm::ArrayRef<uint8_t>(
                              reinterpret_cast<const uint8_t *>(objBytes.data()),
                              objBytes.size()))) {
    result.Timings.llvmMcSeconds +=
        timingElapsed(options.CollectTimings, llvmMcStart);
    return false;
  }
  result.Timings.llvmMcSeconds +=
      timingElapsed(options.CollectTimings, llvmMcStart);

  return true;
}

// Link one or more relocatable .o files into a shared HSACO using the
// in-process LLD ELF driver.
static bool linkObjects(llvm::ArrayRef<std::string> objPaths,
                        llvm::StringRef hsacoPath) {
  std::string hsacoPathStr = hsacoPath.str();
  llvm::SmallVector<const char *, 16> args;
  args.push_back("ld.lld");
  args.push_back("-shared");
  args.push_back("--threads=1");
  args.push_back("-o");
  args.push_back(hsacoPathStr.c_str());
  for (auto &o : objPaths)
    args.push_back(o.c_str());

  lld::Result ret =
      lld::lldMain(args, llvm::outs(), llvm::errs(), {{lld::Gnu, &lld::elf::link}});
  lld::CommonLinkerContext::destroy();
  if (ret.retCode != 0 || !ret.canRunAgain) {
    llvm::errs() << "transpiler: ld.lld failed\n";
    return false;
  }
  return true;
}

void collectTargetPrivateSegmentMetadata(PipelineResult &result,
                                         llvm::ArrayRef<std::string> kernelNames) {
  using namespace llvm::amdhsa;
  if (!result.Hsaco || result.Hsaco->getBufferSize() == 0)
    return;
  llvm::MemoryBufferRef hsacoBuf = result.Hsaco->getMemBufferRef();
  for (llvm::StringRef kernelName : kernelNames) {
    llvm::Expected<KernelMeta> metaOrErr =
        extractKernelMeta(hsacoBuf, kernelName);
    if (!metaOrErr) {
      llvm::consumeError(metaOrErr.takeError());
      continue;
    }
    KernelMeta &meta = *metaOrErr;
    if (!meta.HasKernelDescriptor)
      continue;
    result.TargetPrivateSegmentFixedSize = std::max(
        result.TargetPrivateSegmentFixedSize,
        static_cast<uint32_t>(meta.PrivateSegmentFixedSize));
    const bool enabled =
        (meta.ComputePgmRsrc2 &
         (1u << COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT_SHIFT)) != 0;
    result.TargetEnablePrivateSegment |= enabled;
  }
}

PipelineResult runPipeline(llvm::MemoryBufferRef codeObjectData,
                           llvm::StringRef sourceISA,
                           llvm::StringRef targetISA,
                           llvm::StringRef kernelName,
                           PipelineOptions options) {
  auto totalStart = timingStart(options.CollectTimings);
  PipelineResult result;
  auto finish = [&]() {
    result.Timings.totalSeconds = timingElapsed(options.CollectTimings, totalStart);
    return std::move(result);
  };

  auto extractTextStart = timingStart(options.CollectTimings);
  llvm::Expected<TextSection> textOrErr = extractTextSection(codeObjectData);
  result.Timings.extractTextSeconds =
      timingElapsed(options.CollectTimings, extractTextStart);
  if (!textOrErr) {
    llvm::errs() << "transpiler: Failed to extract .text section: "
                 << llvm::toString(textOrErr.takeError()) << "\n";
    return finish();
  }
  TextSection text = std::move(*textOrErr);

  auto tempDirStart = timingStart(options.CollectTimings);
  DumpDir tmpDir;
  result.Timings.createTempDirSeconds =
      timingElapsed(options.CollectTimings, tempDirStart);
  if (!tmpDir.valid)
    return finish();

  {
    static const char *s_dumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
    if (s_dumpInput && s_dumpInput[0] == '1')
      writeFile(tmpDir.filePath("input.co"),
                llvm::ArrayRef(reinterpret_cast<const uint8_t *>(
                                   codeObjectData.getBufferStart()),
                               codeObjectData.getBufferSize()));
  }

  std::string objPath   = tmpDir.filePath("kernel.o");
  std::string hsacoPath = tmpDir.filePath("kernel.Hsaco");

  if (!raiseAndCompileKernel(text, codeObjectData, kernelName,
                             sourceISA, targetISA, tmpDir, objPath, result,
                             options))
    return finish();

  auto linkStart = timingStart(options.CollectTimings);
  if (!linkObjects({objPath}, hsacoPath))
    return finish();
  result.Timings.linkSeconds += timingElapsed(options.CollectTimings, linkStart);

  auto readHsacoStart = timingStart(options.CollectTimings);
  if (auto hsacoBufOrErr =
          llvm::MemoryBuffer::getFile(hsacoPath, /*IsText=*/false)) {
    result.Hsaco = std::move(*hsacoBufOrErr);
  } else {
    llvm::errs() << "transpiler: Cannot read HSACO: " << hsacoPath << ": "
                 << hsacoBufOrErr.getError().message() << "\n";
  }
  result.Timings.readHsacoSeconds +=
      timingElapsed(options.CollectTimings, readHsacoStart);
  if (!result.Hsaco || result.Hsaco->getBufferSize() == 0) {
    llvm::errs() << "transpiler: Failed to read HSACO\n";
    return finish();
  }
  std::string kernelNameStr = kernelName.str();
  auto collectMetadataStart = timingStart(options.CollectTimings);
  collectTargetPrivateSegmentMetadata(result, {kernelNameStr});
  result.Timings.collectMetadataSeconds +=
      timingElapsed(options.CollectTimings, collectMetadataStart);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: HSACO generated: "
                          << result.Hsaco->getBufferSize() << " bytes\n");
  result.Success = true;
  return finish();
}

PipelineResult runPipelineAllKernels(llvm::MemoryBufferRef codeObjectData,
                                     llvm::StringRef sourceISA,
                                     llvm::StringRef targetISA,
                                     PipelineOptions options) {
  auto totalStart = timingStart(options.CollectTimings);
  PipelineResult result;
  auto finish = [&]() {
    result.Timings.totalSeconds = timingElapsed(options.CollectTimings, totalStart);
    return std::move(result);
  };

  auto listKernelsStart = timingStart(options.CollectTimings);
  llvm::Expected<llvm::SmallVector<std::string>> kernelNamesOrErr =
      listKernelNames(codeObjectData);
  result.Timings.listKernelsSeconds =
      timingElapsed(options.CollectTimings, listKernelsStart);
  if (!kernelNamesOrErr) {
    llvm::errs() << "transpiler: No kernels found in code object: "
                 << llvm::toString(kernelNamesOrErr.takeError()) << "\n";
    return finish();
  }
  llvm::SmallVector<std::string> kernelNames = std::move(*kernelNamesOrErr);
  if (kernelNames.empty()) {
    llvm::errs() << "transpiler: No kernels found in code object\n";
    return finish();
  }

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Raising " << kernelNames.size()
                          << " kernel(s) [" << sourceISA << " -> " << targetISA
                          << "]\n");

  auto extractTextStart = timingStart(options.CollectTimings);
  llvm::Expected<TextSection> textOrErr = extractTextSection(codeObjectData);
  result.Timings.extractTextSeconds =
      timingElapsed(options.CollectTimings, extractTextStart);
  if (!textOrErr) {
    llvm::errs() << "transpiler: Failed to extract .text section: "
                 << llvm::toString(textOrErr.takeError()) << "\n";
    return finish();
  }
  TextSection text = std::move(*textOrErr);

  auto tempDirStart = timingStart(options.CollectTimings);
  DumpDir tmpDir;
  result.Timings.createTempDirSeconds =
      timingElapsed(options.CollectTimings, tempDirStart);
  if (!tmpDir.valid)
    return finish();

  static const char *s_dumpInput = std::getenv("HSA_HOTSWAP_DUMP_INPUT");
  if (s_dumpInput && s_dumpInput[0] == '1')
    writeFile(tmpDir.filePath("input.co"),
              llvm::ArrayRef(reinterpret_cast<const uint8_t *>(
                                 codeObjectData.getBufferStart()),
                             codeObjectData.getBufferSize()));

  std::vector<std::string> objPaths;
  for (size_t i = 0; i < kernelNames.size(); ++i) {
    const auto &kName = kernelNames[i];
    std::string objPath = tmpDir.filePath("k" + std::to_string(i) + ".o");

    LLVM_DEBUG(llvm::dbgs() << "transpiler:   [" << (i + 1) << "/"
                            << kernelNames.size() << "] " << kName << " ... ");

    if (!raiseAndCompileKernel(text, codeObjectData, kName,
                               sourceISA, targetISA, tmpDir, objPath, result,
                               options)) {
      LLVM_DEBUG(llvm::dbgs() << "FAILED\n");
      result.Success = false;
      return finish();
    }
    LLVM_DEBUG(llvm::dbgs() << "OK\n");
    objPaths.push_back(std::move(objPath));
  }

  std::string hsacoPath = tmpDir.filePath("merged.Hsaco");
  auto linkStart = timingStart(options.CollectTimings);
  if (!linkObjects(objPaths, hsacoPath))
    return finish();
  result.Timings.linkSeconds += timingElapsed(options.CollectTimings, linkStart);

  auto readHsacoStart = timingStart(options.CollectTimings);
  if (auto hsacoBufOrErr =
          llvm::MemoryBuffer::getFile(hsacoPath, /*IsText=*/false)) {
    result.Hsaco = std::move(*hsacoBufOrErr);
  } else {
    llvm::errs() << "transpiler: Cannot read HSACO: " << hsacoPath << ": "
                 << hsacoBufOrErr.getError().message() << "\n";
  }
  result.Timings.readHsacoSeconds +=
      timingElapsed(options.CollectTimings, readHsacoStart);
  if (!result.Hsaco || result.Hsaco->getBufferSize() == 0) {
    llvm::errs() << "transpiler: Failed to read merged HSACO\n";
    return finish();
  }
  auto collectMetadataStart = timingStart(options.CollectTimings);
  collectTargetPrivateSegmentMetadata(result, kernelNames);
  result.Timings.collectMetadataSeconds +=
      timingElapsed(options.CollectTimings, collectMetadataStart);

  LLVM_DEBUG(llvm::dbgs() << "transpiler: Merged HSACO: "
                          << result.Hsaco->getBufferSize() << " bytes, "
                          << kernelNames.size() << " kernel(s)\n");
  result.Success = true;
  return finish();
}

} // namespace COMGR::hotswap
