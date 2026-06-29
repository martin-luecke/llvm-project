//===- hsa_tool.cpp - HSA tool half: capture + transpile -----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// HSA_TOOLS_LIB half of the interposer. Hooks every CoreApiTable entry point
/// through which a code-object ELF becomes a loaded executable, captures the
/// spoofed-source (gfx1250) ELF while it is still in host memory, transpiles it
/// to the real device ISA via COMGR, and substitutes the result. Capturing at
/// the load layer (rather than the KFD mapping) is mandatory: that is the only
/// place the transpilable ELF exists -- by the time segments are mapped to the
/// device the metadata note is gone and relocations are applied. Covering all
/// load entry points makes capture complete by construction at that layer.
///
/// The transpile target is the real device detected beneath the KFD spoof by
/// the LD_PRELOAD half, not the (spoofed) agent ISA, because under the spoof
/// the agent ISA reports the source target. It is supplied via
/// HSA_HOTSWAP_TARGET.
///
//===----------------------------------------------------------------------===//

#include "topology_spoof.h"
#include "transpile.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <elf.h>
#include <hsa.h>
#include <hsa_api_trace.h>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace {

using ByteVec = std::shared_ptr<std::vector<uint8_t>>;
using OwnedElf = std::unique_ptr<void, decltype(&std::free)>;

// Captured ELF bytes, keyed by the handle of the object that owns them: a
// code-object reader (modern path) or a hsa_code_object_t (deprecated path).
std::mutex GMapMutex;
std::unordered_map<uint64_t, ByteVec> GReaderMap;
std::unordered_map<uint64_t, ByteVec> GObjMap;

// Rewritten ELFs must outlive the executable: ROCr's LoadedCodeObjectImpl keeps
// a raw pointer into the ELF data. Kept alive until OnUnload (then leaked
// rather than risk a teardown use-after-free).
std::mutex GRewrittenMutex;
std::vector<OwnedElf> GRewritten;

CoreApiTable *GCoreTable = nullptr;
decltype(hsa_code_object_reader_create_from_memory) *GOrigReaderFromMemory =
    nullptr;
decltype(hsa_code_object_reader_create_from_file) *GOrigReaderFromFile =
    nullptr;
decltype(hsa_code_object_reader_destroy) *GOrigReaderDestroy = nullptr;
decltype(hsa_executable_load_agent_code_object) *GOrigLoadAgentCodeObject =
    nullptr;
decltype(hsa_executable_load_program_code_object) *GOrigLoadProgramCodeObject =
    nullptr;
decltype(hsa_executable_load_code_object) *GOrigLoadCodeObject = nullptr;
decltype(hsa_code_object_deserialize) *GOrigCodeObjectDeserialize = nullptr;
decltype(hsa_code_object_destroy) *GOrigCodeObjectDestroy = nullptr;
decltype(hsa_isa_get_info_alt) *GOrigIsaGetInfoAlt = nullptr;
decltype(hsa_agent_iterate_isas) *GOrigAgentIterateIsas = nullptr;

// Diagnostics honour the native HotSwap name (HSA_HOTSWAP_VERBOSE) as well as
// the interposer's own, so behaviour matches the in-runtime integration.
bool logEnabled() {
  static bool Enabled = std::getenv("HOTSWAP_INTERPOSER_LOG") != nullptr ||
                        std::getenv("HSA_HOTSWAP_VERBOSE") != nullptr;
  return Enabled;
}

// Mirror the native runtime's supportability switch: when set, the load layer
// forwards every code object untouched (no transpile).
bool hotswapDisabled() {
  static bool Disabled = std::getenv("HSA_HOTSWAP_DISABLE") != nullptr;
  return Disabled;
}

/// Optional debug: dump captured/transpiled code objects to this directory.
const char *dumpDir() {
  static const char *Dir = std::getenv("HOTSWAP_INTERPOSER_DUMP_DIR");
  return Dir;
}

void dumpObject(const char *Tag, const void *Data, size_t Size) {
  const char *Dir = dumpDir();
  if (!Dir)
    return;
  static std::atomic<unsigned> Seq{0};
  char Path[4096];
  std::snprintf(Path, sizeof(Path), "%s/co_%u_%s.co", Dir, Seq.fetch_add(1),
                Tag);
  if (FILE *F = std::fopen(Path, "wb")) {
    (void)std::fwrite(Data, 1, Size, F);
    std::fclose(F);
  }
}

// Processor-selection mask for the AMDGPU e_flags field (EF_AMDGPU_MACH).
constexpr uint32_t EfAmdgpuMachMask = 0xff;

// AMDGCN processor value -> gfx target name, mirroring the AMDGPU_MACH_LIST in
// llvm/BinaryFormat/ELF.h (replicated so the tool depends only on <elf.h>;
// these are append-only ABI values, refresh from ELF.h for newer GPUs). The
// list must stay complete -- including the *-generic targets -- because a code
// object whose mach is unrecognised is treated as device-independent and
// forwarded untouched, which on a real device would let a foreign-ISA object
// reach the hardware untranspiled.
#define HOTSWAP_AMDGCN_MACH_LIST(X)                                            \
  X(0x2c, "gfx900")                                                            \
  X(0x2d, "gfx902")                                                            \
  X(0x2e, "gfx904")                                                            \
  X(0x2f, "gfx906")                                                            \
  X(0x30, "gfx908")                                                            \
  X(0x31, "gfx909")                                                            \
  X(0x32, "gfx90c")                                                            \
  X(0x3f, "gfx90a")                                                            \
  X(0x4c, "gfx942")                                                            \
  X(0x4f, "gfx950")                                                            \
  X(0x33, "gfx1010")                                                           \
  X(0x34, "gfx1011")                                                           \
  X(0x35, "gfx1012")                                                           \
  X(0x42, "gfx1013")                                                           \
  X(0x36, "gfx1030")                                                           \
  X(0x37, "gfx1031")                                                           \
  X(0x38, "gfx1032")                                                           \
  X(0x39, "gfx1033")                                                           \
  X(0x3e, "gfx1034")                                                           \
  X(0x3d, "gfx1035")                                                           \
  X(0x45, "gfx1036")                                                           \
  X(0x41, "gfx1100")                                                           \
  X(0x46, "gfx1101")                                                           \
  X(0x47, "gfx1102")                                                           \
  X(0x44, "gfx1103")                                                           \
  X(0x43, "gfx1150")                                                           \
  X(0x4a, "gfx1151")                                                           \
  X(0x55, "gfx1152")                                                           \
  X(0x58, "gfx1153")                                                           \
  X(0x5d, "gfx1170")                                                           \
  X(0x5e, "gfx1171")                                                           \
  X(0x5c, "gfx1172")                                                           \
  X(0x48, "gfx1200")                                                           \
  X(0x4e, "gfx1201")                                                           \
  X(0x49, "gfx1250")                                                           \
  X(0x5a, "gfx1251")                                                           \
  X(0x50, "gfx1310")                                                           \
  X(0x51, "gfx9-generic")                                                      \
  X(0x52, "gfx10-1-generic")                                                   \
  X(0x53, "gfx10-3-generic")                                                   \
  X(0x54, "gfx11-generic")                                                     \
  X(0x59, "gfx12-generic")                                                     \
  X(0x5b, "gfx12-5-generic")                                                   \
  X(0x5f, "gfx9-4-generic")

std::string gfxTargetFromMach(uint32_t Mach) {
  switch (Mach) {
#define HOTSWAP_MACH_CASE(NUM, NAME)                                           \
  case NUM:                                                                    \
    return NAME;
    HOTSWAP_AMDGCN_MACH_LIST(HOTSWAP_MACH_CASE)
#undef HOTSWAP_MACH_CASE
  default:
    return {};
  }
}

const Elf64_Ehdr *validateElf64(const uint8_t *Elf, size_t Size) {
  if (Size < sizeof(Elf64_Ehdr))
    return nullptr;
  const auto *Ehdr = reinterpret_cast<const Elf64_Ehdr *>(Elf);
  if (std::memcmp(Ehdr->e_ident, ELFMAG, SELFMAG) != 0)
    return nullptr;
  if (Ehdr->e_ident[EI_CLASS] != ELFCLASS64)
    return nullptr;
  return Ehdr;
}

/// Read the code object's source ISA from the ELF e_flags EF_AMDGPU_MACH field.
std::string readElfIsa(const uint8_t *Elf, size_t Size) {
  if (const Elf64_Ehdr *Ehdr = validateElf64(Elf, Size)) {
    std::string Gfx = gfxTargetFromMach(Ehdr->e_flags & EfAmdgpuMachMask);
    if (!Gfx.empty())
      return "amdgcn-amd-amdhsa--" + Gfx;
  }
  return {};
}

std::string extractGfxName(const std::string &Isa) {
  constexpr const char Prefix[] = "amdgcn-amd-amdhsa--";
  std::string Target = Isa;
  if (Target.rfind(Prefix, 0) == 0)
    Target.erase(0, sizeof(Prefix) - 1);
  size_t Colon = Target.find(':');
  if (Colon != std::string::npos)
    Target.resize(Colon);
  if (Target.rfind("gfx", 0) != 0 || Target.size() <= 3)
    return {};
  return Target;
}

/// Resolve the real device ISA via the original (un-hooked) entry points. All
/// HSA calls go through CoreApiTable pointers, never direct hsa_* symbols, so
/// the library carries no undefined HSA symbols when LD_PRELOAD'd ahead of
/// libhsa-runtime.
std::string agentIsaName(hsa_agent_t Agent) {
  if (!GOrigAgentIterateIsas || !GOrigIsaGetInfoAlt)
    return {};
  auto Cb = [](hsa_isa_t Isa, void *Data) -> hsa_status_t {
    auto *Name = static_cast<std::string *>(Data);
    uint32_t Len = 0;
    if (GOrigIsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME_LENGTH, &Len) !=
        HSA_STATUS_SUCCESS)
      return HSA_STATUS_ERROR;
    Name->resize(Len);
    if (GOrigIsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME, Name->data()) !=
        HSA_STATUS_SUCCESS) {
      Name->clear();
      return HSA_STATUS_ERROR;
    }
    if (!Name->empty() && Name->back() == '\0')
      Name->pop_back();
    return HSA_STATUS_INFO_BREAK;
  };
  std::string Name;
  GOrigAgentIterateIsas(Agent, Cb, &Name);
  return Name;
}

/// The transpile target: the real device beneath the spoof. Priority:
/// HSA_HOTSWAP_TARGET (the convergence interface with the native runtime; the
/// LD_PRELOAD half publishes the detected real device here), then the
/// in-process detected real gfx, then (only when an agent is available) the
/// agent ISA. Under the spoof the agent ISA is the spoofed source, so it is the
/// last resort -- the env/detected value is what names the real device.
std::string resolveTargetIsa(const hsa_agent_t *Agent) {
  if (const char *Env = std::getenv("HSA_HOTSWAP_TARGET")) {
    std::string Gfx = extractGfxName(Env);
    if (!Gfx.empty())
      return "amdgcn-amd-amdhsa--" + Gfx;
  }
  if (uint32_t Real = hotswap::interposer::detectedRealGfxVersion())
    return "amdgcn-amd-amdhsa--" +
           hotswap::interposer::gfxTargetVersionName(Real);
  if (Agent)
    return agentIsaName(*Agent);
  return {};
}

ByteVec lookupReader(uint64_t Handle) {
  std::scoped_lock Lock(GMapMutex);
  auto It = GReaderMap.find(Handle);
  return It != GReaderMap.end() ? It->second : ByteVec{};
}

ByteVec lookupObj(uint64_t Handle) {
  std::scoped_lock Lock(GMapMutex);
  auto It = GObjMap.find(Handle);
  return It != GObjMap.end() ? It->second : ByteVec{};
}

void retainRewritten(OwnedElf Elf) {
  try {
    std::scoped_lock Lock(GRewrittenMutex);
    GRewritten.push_back(std::move(Elf));
  } catch (const std::bad_alloc &) {
    (void)Elf.release();
  }
}

enum class Decision { Passthrough, Transpiled, Refuse };

/// Shared transpile decision for every load entry point. Returns Passthrough
/// for objects already built for the real device (load the original untouched),
/// Transpiled with a malloc'd OutElf for foreign-ISA (spoofed-source) objects,
/// or Refuse (fail closed) when the source/target ISA cannot be determined.
Decision decideAndTranspile(const ByteVec &Bytes, const hsa_agent_t *Agent,
                            void **OutElf, size_t *OutSize) {
  *OutElf = nullptr;
  *OutSize = 0;
  if (hotswapDisabled())
    return Decision::Passthrough;
  std::string SourceIsa = readElfIsa(Bytes->data(), Bytes->size());
  std::string TargetIsa = resolveTargetIsa(Agent);
  std::string SourceGfx = extractGfxName(SourceIsa);
  std::string TargetGfx = extractGfxName(TargetIsa);

  // A code object with no AMDGPU ISA (e.g. a program-scope object that only
  // carries variables) is device-independent; load it untouched.
  if (SourceGfx.empty())
    return Decision::Passthrough;

  if (TargetGfx.empty()) {
    std::fprintf(
        stderr,
        "[hotswap-interposer] no transpile target for source %s; refusing\n",
        SourceIsa.c_str());
    return Decision::Refuse;
  }

  // Already built for the real device (e.g. CLR's native builtin shaders).
  if (SourceGfx == TargetGfx)
    return Decision::Passthrough;

  dumpObject(SourceGfx.c_str(), Bytes->data(), Bytes->size());

  int Rc = hotswap::interposer::retargetCodeObject(
      Bytes->data(), Bytes->size(), SourceIsa.c_str(), TargetIsa.c_str(),
      OutElf, OutSize);
  if (Rc != 0 || !*OutElf || *OutSize == 0) {
    std::fprintf(
        stderr,
        "[hotswap-interposer] transpile %s -> %s failed (rc=%d); refusing\n",
        SourceIsa.c_str(), TargetIsa.c_str(), Rc);
    return Decision::Refuse;
  }
  if (logEnabled())
    std::fprintf(stderr,
                 "[hotswap-interposer] transpiled %s -> %s (%zu bytes)\n",
                 SourceIsa.c_str(), TargetIsa.c_str(), *OutSize);
  dumpObject(TargetGfx.c_str(), *OutElf, *OutSize);
  return Decision::Transpiled;
}

// -- reader capture (modern path) --

hsa_status_t HSA_API hookReaderFromMemory(const void *CodeObject, size_t Size,
                                          hsa_code_object_reader_t *Reader) {
  hsa_code_object_reader_t R = {};
  hsa_status_t St = GOrigReaderFromMemory(CodeObject, Size, &R);
  if (St != HSA_STATUS_SUCCESS)
    return St;
  try {
    auto Vec = std::make_shared<std::vector<uint8_t>>(
        static_cast<const uint8_t *>(CodeObject),
        static_cast<const uint8_t *>(CodeObject) + Size);
    std::scoped_lock Lock(GMapMutex);
    GReaderMap[R.handle] = std::move(Vec);
  } catch (const std::bad_alloc &) {
    GOrigReaderDestroy(R);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  *Reader = R;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API hookReaderFromFile(hsa_file_t File,
                                        hsa_code_object_reader_t *Reader) {
  // The caller retains ownership of File and may reuse it, so its read offset
  // is saved and restored around the slurp. (Converting a file reader to a
  // memory reader loses the URI provenance the file path would otherwise
  // carry.)
  off_t SavedPos = ::lseek(File, 0, SEEK_CUR);
  off_t End = ::lseek(File, 0, SEEK_END);
  if (SavedPos < 0 || End < 0)
    return HSA_STATUS_ERROR_INVALID_FILE;
  ::lseek(File, 0, SEEK_SET);
  hsa_status_t Result = HSA_STATUS_ERROR_INVALID_FILE;
  try {
    auto Vec = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(End));
    size_t Got = 0;
    bool ReadOk = true;
    while (Got < Vec->size()) {
      ssize_t N = ::read(File, Vec->data() + Got, Vec->size() - Got);
      if (N <= 0) {
        ReadOk = false;
        break;
      }
      Got += static_cast<size_t>(N);
    }
    if (ReadOk) {
      hsa_code_object_reader_t R = {};
      hsa_status_t St = GOrigReaderFromMemory(Vec->data(), Vec->size(), &R);
      if (St == HSA_STATUS_SUCCESS) {
        {
          std::scoped_lock Lock(GMapMutex);
          GReaderMap[R.handle] = std::move(Vec);
        }
        *Reader = R;
        Result = HSA_STATUS_SUCCESS;
      } else {
        Result = St;
      }
    }
  } catch (const std::bad_alloc &) {
    Result = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  ::lseek(File, SavedPos, SEEK_SET);
  return Result;
}

hsa_status_t HSA_API hookReaderDestroy(hsa_code_object_reader_t Reader) {
  {
    std::scoped_lock Lock(GMapMutex);
    GReaderMap.erase(Reader.handle);
  }
  return GOrigReaderDestroy(Reader);
}

// -- agent-scoped load (modern path) --

hsa_status_t loadRewrittenReader(hsa_executable_t Exec, hsa_agent_t Agent,
                                 const char *Options,
                                 hsa_loaded_code_object_t *Loaded, void *OutElf,
                                 size_t OutSize) {
  OwnedElf Owned(OutElf, &std::free);
  hsa_code_object_reader_t NewReader = {};
  hsa_status_t St = GOrigReaderFromMemory(Owned.get(), OutSize, &NewReader);
  if (St != HSA_STATUS_SUCCESS)
    return St;
  St = GOrigLoadAgentCodeObject(Exec, Agent, NewReader, Options, Loaded);
  GOrigReaderDestroy(NewReader);
  if (St == HSA_STATUS_SUCCESS)
    retainRewritten(std::move(Owned));
  return St;
}

hsa_status_t HSA_API hookLoadAgentCodeObject(hsa_executable_t Exec,
                                             hsa_agent_t Agent,
                                             hsa_code_object_reader_t Reader,
                                             const char *Options,
                                             hsa_loaded_code_object_t *Loaded) {
  ByteVec Bytes = lookupReader(Reader.handle);
  if (!Bytes) {
    std::fprintf(stderr,
                 "[hotswap-interposer] no staged bytes for reader; refusing\n");
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  }
  void *OutElf = nullptr;
  size_t OutSize = 0;
  switch (decideAndTranspile(Bytes, &Agent, &OutElf, &OutSize)) {
  case Decision::Passthrough:
    return GOrigLoadAgentCodeObject(Exec, Agent, Reader, Options, Loaded);
  case Decision::Transpiled:
    return loadRewrittenReader(Exec, Agent, Options, Loaded, OutElf, OutSize);
  case Decision::Refuse:
  default:
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
}

// -- program-scope load (modern path; no agent) --

hsa_status_t loadRewrittenProgram(hsa_executable_t Exec, const char *Options,
                                  hsa_loaded_code_object_t *Loaded,
                                  void *OutElf, size_t OutSize) {
  OwnedElf Owned(OutElf, &std::free);
  hsa_code_object_reader_t NewReader = {};
  hsa_status_t St = GOrigReaderFromMemory(Owned.get(), OutSize, &NewReader);
  if (St != HSA_STATUS_SUCCESS)
    return St;
  St = GOrigLoadProgramCodeObject(Exec, NewReader, Options, Loaded);
  GOrigReaderDestroy(NewReader);
  if (St == HSA_STATUS_SUCCESS)
    retainRewritten(std::move(Owned));
  return St;
}

hsa_status_t HSA_API hookLoadProgramCodeObject(
    hsa_executable_t Exec, hsa_code_object_reader_t Reader, const char *Options,
    hsa_loaded_code_object_t *Loaded) {
  ByteVec Bytes = lookupReader(Reader.handle);
  if (!Bytes)
    return GOrigLoadProgramCodeObject(Exec, Reader, Options, Loaded);
  void *OutElf = nullptr;
  size_t OutSize = 0;
  switch (decideAndTranspile(Bytes, nullptr, &OutElf, &OutSize)) {
  case Decision::Passthrough:
    return GOrigLoadProgramCodeObject(Exec, Reader, Options, Loaded);
  case Decision::Transpiled:
    return loadRewrittenProgram(Exec, Options, Loaded, OutElf, OutSize);
  case Decision::Refuse:
  default:
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
}

// -- deprecated hsa_code_object_t path --

hsa_status_t HSA_API hookCodeObjectDeserialize(void *Serialized, size_t Size,
                                               const char *Options,
                                               hsa_code_object_t *CodeObject) {
  hsa_status_t St =
      GOrigCodeObjectDeserialize(Serialized, Size, Options, CodeObject);
  if (St != HSA_STATUS_SUCCESS)
    return St;
  try {
    auto Vec = std::make_shared<std::vector<uint8_t>>(
        static_cast<const uint8_t *>(Serialized),
        static_cast<const uint8_t *>(Serialized) + Size);
    std::scoped_lock Lock(GMapMutex);
    GObjMap[CodeObject->handle] = std::move(Vec);
  } catch (const std::bad_alloc &) {
    // Capture failed; the deprecated load hook will fail closed on the miss.
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API hookCodeObjectDestroy(hsa_code_object_t CodeObject) {
  {
    std::scoped_lock Lock(GMapMutex);
    GObjMap.erase(CodeObject.handle);
  }
  return GOrigCodeObjectDestroy(CodeObject);
}

hsa_status_t HSA_API hookLoadCodeObject(hsa_executable_t Exec,
                                        hsa_agent_t Agent,
                                        hsa_code_object_t CodeObject,
                                        const char *Options) {
  ByteVec Bytes = lookupObj(CodeObject.handle);
  if (!Bytes) {
    std::fprintf(
        stderr,
        "[hotswap-interposer] no staged bytes for code object; refusing\n");
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  void *OutElf = nullptr;
  size_t OutSize = 0;
  switch (decideAndTranspile(Bytes, &Agent, &OutElf, &OutSize)) {
  case Decision::Passthrough:
    return GOrigLoadCodeObject(Exec, Agent, CodeObject, Options);
  case Decision::Transpiled: {
    OwnedElf Owned(OutElf, &std::free);
    hsa_code_object_t NewObj = {};
    hsa_status_t St =
        GOrigCodeObjectDeserialize(Owned.get(), OutSize, Options, &NewObj);
    if (St != HSA_STATUS_SUCCESS)
      return St;
    St = GOrigLoadCodeObject(Exec, Agent, NewObj, Options);
    if (St == HSA_STATUS_SUCCESS)
      retainRewritten(std::move(Owned));
    return St;
  }
  case Decision::Refuse:
  default:
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
}

} // namespace

extern "C" {

HOTSWAP_EXPORT
bool OnLoad(HsaApiTable *Table, uint64_t RuntimeVersion, uint64_t FailedCount,
            const char *const *FailedNames) {
  (void)RuntimeVersion;
  (void)FailedCount;
  (void)FailedNames;
  if (!Table || !Table->core_)
    return false;
  CoreApiTable *Core = Table->core_;
  if (!Core->hsa_code_object_reader_create_from_memory_fn ||
      !Core->hsa_code_object_reader_create_from_file_fn ||
      !Core->hsa_code_object_reader_destroy_fn ||
      !Core->hsa_executable_load_agent_code_object_fn)
    return false;

  GCoreTable = Core;
  GOrigReaderFromMemory = Core->hsa_code_object_reader_create_from_memory_fn;
  GOrigReaderFromFile = Core->hsa_code_object_reader_create_from_file_fn;
  GOrigReaderDestroy = Core->hsa_code_object_reader_destroy_fn;
  GOrigLoadAgentCodeObject = Core->hsa_executable_load_agent_code_object_fn;
  GOrigLoadProgramCodeObject = Core->hsa_executable_load_program_code_object_fn;
  GOrigLoadCodeObject = Core->hsa_executable_load_code_object_fn;
  GOrigCodeObjectDeserialize = Core->hsa_code_object_deserialize_fn;
  GOrigCodeObjectDestroy = Core->hsa_code_object_destroy_fn;
  GOrigIsaGetInfoAlt = Core->hsa_isa_get_info_alt_fn;
  GOrigAgentIterateIsas = Core->hsa_agent_iterate_isas_fn;

  Core->hsa_code_object_reader_create_from_memory_fn = hookReaderFromMemory;
  Core->hsa_code_object_reader_create_from_file_fn = hookReaderFromFile;
  Core->hsa_code_object_reader_destroy_fn = hookReaderDestroy;
  Core->hsa_executable_load_agent_code_object_fn = hookLoadAgentCodeObject;
  // Close the remaining load-path holes so capture is complete at the load
  // layer.
  if (GOrigLoadProgramCodeObject)
    Core->hsa_executable_load_program_code_object_fn =
        hookLoadProgramCodeObject;
  if (GOrigCodeObjectDeserialize)
    Core->hsa_code_object_deserialize_fn = hookCodeObjectDeserialize;
  if (GOrigCodeObjectDestroy)
    Core->hsa_code_object_destroy_fn = hookCodeObjectDestroy;
  if (GOrigLoadCodeObject && GOrigCodeObjectDeserialize)
    Core->hsa_executable_load_code_object_fn = hookLoadCodeObject;

  if (logEnabled())
    std::fprintf(
        stderr,
        "[hotswap-interposer] HSA tool loaded; capturing code objects\n");
  return true;
}

HOTSWAP_EXPORT
void OnUnload() {
  if (GCoreTable) {
    GCoreTable->hsa_code_object_reader_create_from_memory_fn =
        GOrigReaderFromMemory;
    GCoreTable->hsa_code_object_reader_create_from_file_fn =
        GOrigReaderFromFile;
    GCoreTable->hsa_code_object_reader_destroy_fn = GOrigReaderDestroy;
    GCoreTable->hsa_executable_load_agent_code_object_fn =
        GOrigLoadAgentCodeObject;
    if (GOrigLoadProgramCodeObject)
      GCoreTable->hsa_executable_load_program_code_object_fn =
          GOrigLoadProgramCodeObject;
    if (GOrigCodeObjectDeserialize)
      GCoreTable->hsa_code_object_deserialize_fn = GOrigCodeObjectDeserialize;
    if (GOrigCodeObjectDestroy)
      GCoreTable->hsa_code_object_destroy_fn = GOrigCodeObjectDestroy;
    if (GOrigLoadCodeObject && GOrigCodeObjectDeserialize)
      GCoreTable->hsa_executable_load_code_object_fn = GOrigLoadCodeObject;
    GCoreTable = nullptr;
  }
  {
    std::scoped_lock Lock(GMapMutex);
    GReaderMap.clear();
    GObjMap.clear();
  }
  {
    std::scoped_lock Lock(GRewrittenMutex);
    // ROCr/ROCclr teardown may still hold raw pointers into rewritten ELFs;
    // leak at process exit rather than risk a use-after-free.
    for (auto &Elf : GRewritten)
      (void)Elf.release();
    GRewritten.clear();
  }
}

} // extern "C"
