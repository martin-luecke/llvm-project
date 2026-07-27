//===- hotswap_tool.cpp - COMGR-backed HSA presentation tool -------------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prototype HSA API tool for cross-ISA presentation. ROCr remains
// physical-only. The tool presents a logical ISA, loads target-tagged source
// descriptors, and replaces every application dispatch on a proxy queue with
// a lazily translated target kernel.
//
//===----------------------------------------------------------------------===//

#include "hotswap_platform_io.hpp"

#include <amd_comgr.h>
#include <elf.h>
#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ext_amd.h>
#include <hsa_ven_amd_loader.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#define HSA_HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace {

namespace io = hotswap::hsa_tool::platform_io;
using Bytes = std::shared_ptr<std::vector<uint8_t>>;

constexpr uint64_t VirtualIsaTag = 0x4853574100000000ULL;
constexpr uint64_t VirtualWaveTag = 0x4853574200000000ULL;
constexpr uint64_t VirtualTagMask = 0xffffffff00000000ULL;
constexpr uint32_t MachMask = 0xff;
constexpr unsigned SttAmdGpuHsaKernel = 10;

static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64,
              "the interceptor requires 64-byte AQL packets");
static_assert(sizeof(hsa_amd_ext_kernel_dispatch_packet_t) == 64,
              "the interceptor requires 64-byte extended AQL packets");
static_assert(sizeof(hsa_amd_barrier_value_packet_t) == 64,
              "the interceptor requires 64-byte barrier AQL packets");

struct IsaView {
  hsa_agent_t Agent{};
  hsa_isa_t Physical{};
  hsa_isa_t Presented{};
  hsa_wavefront_t Wave{};
  std::string PhysicalName;
  std::string PresentedName;
};

struct LazyObject {
  Bytes Source;
  hsa_agent_t Agent{};
  hsa_executable_t Parent{};
  std::string SourceIsa;
  std::string TargetIsa;
  std::string SourceGfx;
  std::string TargetGfx;
};

struct KernelRecord {
  std::mutex Mutex;
  std::shared_ptr<LazyObject> Object;
  std::string Name;
  uint64_t SourceObject = 0;
  uint64_t TargetObject = 0;
  uint32_t SourcePrivate = 0;
  uint32_t SourceGroup = 0;
  uint32_t TargetPrivate = 0;
  uint32_t TargetGroup = 0;
  uint32_t Scale = 1;
  bool Attempted = false;
  bool Succeeded = false;
  std::string Failure;
};

struct SymbolRecord {
  std::shared_ptr<LazyObject> Object;
  std::string Name;
};

struct ChildRecord {
  hsa_executable_t Parent{};
  hsa_executable_t Executable{};
  Bytes Elf;
};

CoreApiTable *Core;
AmdExtTable *Amd;

#define SAVE_CORE(Name) decltype(Name) *Next_##Name
SAVE_CORE(hsa_iterate_agents);
SAVE_CORE(hsa_agent_get_info);
SAVE_CORE(hsa_isa_from_name);
SAVE_CORE(hsa_agent_iterate_isas);
SAVE_CORE(hsa_isa_get_info);
SAVE_CORE(hsa_isa_get_info_alt);
SAVE_CORE(hsa_isa_compatible);
SAVE_CORE(hsa_isa_get_exception_policies);
SAVE_CORE(hsa_isa_get_round_method);
SAVE_CORE(hsa_isa_iterate_wavefronts);
SAVE_CORE(hsa_wavefront_get_info);
SAVE_CORE(hsa_system_get_extension_table);
SAVE_CORE(hsa_system_get_major_extension_table);
SAVE_CORE(hsa_queue_create);
SAVE_CORE(hsa_queue_destroy);
SAVE_CORE(hsa_code_object_reader_create_from_memory);
SAVE_CORE(hsa_code_object_reader_create_from_file);
SAVE_CORE(hsa_code_object_reader_destroy);
SAVE_CORE(hsa_executable_destroy);
SAVE_CORE(hsa_executable_load_agent_code_object);
SAVE_CORE(hsa_executable_get_symbol_by_name);
SAVE_CORE(hsa_executable_get_symbol);
SAVE_CORE(hsa_executable_symbol_get_info);
SAVE_CORE(hsa_executable_iterate_symbols);
SAVE_CORE(hsa_executable_iterate_agent_symbols);
SAVE_CORE(hsa_executable_iterate_program_symbols);
#undef SAVE_CORE

decltype(hsa_amd_queue_intercept_create) *NextInterceptCreate;
decltype(hsa_amd_queue_intercept_register) *NextInterceptRegister;
decltype(hsa_amd_queue_get_info) *NextQueueGetInfo;
decltype(hsa_amd_queue_create) *NextAmdQueueCreate;
decltype(hsa_amd_queue_set_priority) *NextQueueSetPriority;
decltype(hsa_amd_queue_cu_set_mask) *NextQueueSetCuMask;

// Keep prototype state alive until ROCr calls OnUnload. Process-exit DSO
// destructor order is not a tool lifecycle contract: a tool loaded after ROCr
// can have its C++ destructors run before ROCr's own final shutdown callback.
std::mutex &StateMutex = *new std::mutex;
std::mutex &ProofMutex = *new std::mutex;
std::unordered_map<uint64_t, IsaView> &ViewsByAgent =
    *new std::unordered_map<uint64_t, IsaView>;
std::unordered_map<uint64_t, IsaView> &ViewsByIsa =
    *new std::unordered_map<uint64_t, IsaView>;
std::unordered_map<uint64_t, IsaView> &ViewsByWave =
    *new std::unordered_map<uint64_t, IsaView>;
std::unordered_map<uint64_t, Bytes> &Readers =
    *new std::unordered_map<uint64_t, Bytes>;
auto &ObjectsByExecutable = *new std::unordered_map<
    uint64_t, std::unordered_map<std::string, std::shared_ptr<LazyObject>>>;
std::unordered_map<uint64_t, SymbolRecord> &Symbols =
    *new std::unordered_map<uint64_t, SymbolRecord>;
std::unordered_map<uint64_t, std::shared_ptr<KernelRecord>> &Kernels =
    *new std::unordered_map<uint64_t, std::shared_ptr<KernelRecord>>;
std::unordered_set<const hsa_queue_t *> &ProtectedQueues =
    *new std::unordered_set<const hsa_queue_t *>;
std::vector<Bytes> &SkeletonStorage = *new std::vector<Bytes>;
std::vector<ChildRecord> &Children = *new std::vector<ChildRecord>;

std::string &PresentedGfx = *new std::string;
std::string &CacheDir = *new std::string;
std::string &ProofPath = *new std::string;
bool AssumeHipGlobalOffsetZero;
std::atomic<uint64_t> RegisteredObjectCount{0};
std::atomic<uint64_t> SuccessfulTranslationCount{0};
std::atomic<uint64_t> RewrittenDispatchCount{0};
std::atomic<uint64_t> ProtectedQueueCount{0};
std::atomic<uint64_t> RejectedObjectCount{0};

[[noreturn]] void Refuse(const std::string &Reason) {
  std::fprintf(stderr, "hotswap-tool: refusing execution: %s\n",
               Reason.c_str());
  std::fflush(stderr);
  std::abort();
}

std::string JsonEscape(const std::string &Value) {
  std::string Out;
  for (char C : Value) {
    if (C == '\\')
      Out += "\\\\";
    else if (C == '"')
      Out += "\\\"";
    else if (C == '\n')
      Out += "\\n";
    else
      Out += C;
  }
  return Out;
}

void Proof(const std::string &Fields) {
  if (ProofPath.empty())
    return;
  std::lock_guard<std::mutex> Lock(ProofMutex);
  FILE *Out = std::fopen(ProofPath.c_str(), "a");
  if (!Out) {
    std::fprintf(stderr, "hotswap-tool: cannot open proof log %s: %s\n",
                 ProofPath.c_str(), std::strerror(errno));
    return;
  }
  std::fprintf(Out, "{%s}\n", Fields.c_str());
  std::fclose(Out);
}

bool InRange(size_t Offset, size_t Count, size_t Size) {
  return Offset <= Size && Count <= Size - Offset;
}

const Elf64_Ehdr *GetElf(const uint8_t *Data, size_t Size) {
  if (!Data || !InRange(0, sizeof(Elf64_Ehdr), Size))
    return nullptr;
  const auto *Header = reinterpret_cast<const Elf64_Ehdr *>(Data);
  if (std::memcmp(Header->e_ident, ELFMAG, SELFMAG) != 0 ||
      Header->e_ident[EI_CLASS] != ELFCLASS64 ||
      Header->e_ident[EI_DATA] != ELFDATA2LSB || Header->e_machine != EM_AMDGPU)
    return nullptr;
  return Header;
}

std::string GfxFromMach(uint32_t Mach) {
  switch (Mach & MachMask) {
  case 0x4c:
    return "gfx942";
  case 0x4f:
    return "gfx950";
  case 0x49:
    return "gfx1250";
  default:
    return {};
  }
}

uint32_t MachFromGfx(const std::string &Gfx) {
  if (Gfx == "gfx942")
    return 0x4c;
  if (Gfx == "gfx950")
    return 0x4f;
  if (Gfx == "gfx1250")
    return 0x49;
  return 0;
}

std::string Processor(const std::string &Isa) {
  size_t Begin = Isa.find("gfx");
  if (Begin == std::string::npos)
    return {};
  size_t End = Begin + 3;
  while (End < Isa.size()) {
    char C = Isa[End];
    if (!((C >= '0' && C <= '9') || (C >= 'a' && C <= 'z') ||
          (C >= 'A' && C <= 'Z')))
      break;
    ++End;
  }
  return Isa.substr(Begin, End - Begin);
}

std::string KernelName(std::string Name) {
  if (Name.size() > 3 && Name.compare(Name.size() - 3, 3, ".kd") == 0)
    Name.resize(Name.size() - 3);
  return Name;
}

bool ListKernels(const Bytes &Elf, std::vector<std::string> &Names,
                 std::string &Failure) {
  const Elf64_Ehdr *Header = GetElf(Elf->data(), Elf->size());
  if (!Header || Header->e_shentsize < sizeof(Elf64_Shdr) ||
      Header->e_shnum == 0 ||
      !InRange(Header->e_shoff,
               static_cast<size_t>(Header->e_shentsize) * Header->e_shnum,
               Elf->size())) {
    Failure = "invalid or sectionless AMDGPU ELF";
    return false;
  }
  auto Section = [&](uint16_t Index) -> const Elf64_Shdr * {
    if (Index >= Header->e_shnum)
      return nullptr;
    return reinterpret_cast<const Elf64_Shdr *>(Elf->data() + Header->e_shoff +
                                                static_cast<size_t>(Index) *
                                                    Header->e_shentsize);
  };
  std::unordered_set<std::string> Unique;
  for (uint16_t I = 0; I < Header->e_shnum; ++I) {
    const Elf64_Shdr *Symtab = Section(I);
    if (!Symtab ||
        (Symtab->sh_type != SHT_SYMTAB && Symtab->sh_type != SHT_DYNSYM))
      continue;
    const Elf64_Shdr *Strtab = Section(Symtab->sh_link);
    if (!Strtab || Symtab->sh_entsize < sizeof(Elf64_Sym) ||
        !InRange(Symtab->sh_offset, Symtab->sh_size, Elf->size()) ||
        !InRange(Strtab->sh_offset, Strtab->sh_size, Elf->size())) {
      Failure = "out-of-bounds ELF symbol table";
      return false;
    }
    const char *Strings =
        reinterpret_cast<const char *>(Elf->data() + Strtab->sh_offset);
    size_t Count = Symtab->sh_size / Symtab->sh_entsize;
    for (size_t J = 0; J < Count; ++J) {
      const auto *Symbol = reinterpret_cast<const Elf64_Sym *>(
          Elf->data() + Symtab->sh_offset + J * Symtab->sh_entsize);
      if (Symbol->st_name >= Strtab->sh_size)
        continue;
      const char *Text = Strings + Symbol->st_name;
      const void *End = std::memchr(Text, 0, Strtab->sh_size - Symbol->st_name);
      if (!End || !Text[0])
        continue;
      std::string Name(Text, static_cast<const char *>(End) - Text);
      bool Descriptor =
          Name.size() > 3 && Name.compare(Name.size() - 3, 3, ".kd") == 0;
      if (ELF64_ST_TYPE(Symbol->st_info) == SttAmdGpuHsaKernel || Descriptor)
        Unique.insert(KernelName(std::move(Name)));
    }
  }
  Names.assign(Unique.begin(), Unique.end());
  std::sort(Names.begin(), Names.end());
  if (Names.empty()) {
    Failure = "ELF has no discoverable kernel symbols";
    return false;
  }
  return true;
}

Bytes MakeSkeleton(const Bytes &Source, const std::string &Target,
                   std::string &Failure) {
  const Elf64_Ehdr *Header = GetElf(Source->data(), Source->size());
  uint32_t Mach = MachFromGfx(Target);
  if (!Header || !Mach) {
    Failure = "unsupported skeleton source or target";
    return {};
  }
  Bytes Result = std::make_shared<std::vector<uint8_t>>(*Source);
  auto *Mutable = reinterpret_cast<Elf64_Ehdr *>(Result->data());
  Mutable->e_flags = (Mutable->e_flags & ~MachMask) | Mach;
  return Result;
}

std::string PhysicalIsaName(hsa_isa_t Isa) {
  uint32_t Length = 0;
  if (Next_hsa_isa_get_info_alt(Isa, HSA_ISA_INFO_NAME_LENGTH, &Length) !=
          HSA_STATUS_SUCCESS ||
      !Length)
    return {};
  std::vector<char> Name(Length + 1, 0);
  if (Next_hsa_isa_get_info_alt(Isa, HSA_ISA_INFO_NAME, Name.data()) !=
      HSA_STATUS_SUCCESS)
    return {};
  return Name.data();
}

bool IsGpu(hsa_agent_t Agent) {
  hsa_device_type_t Type = HSA_DEVICE_TYPE_CPU;
  return Next_hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &Type) ==
             HSA_STATUS_SUCCESS &&
         Type == HSA_DEVICE_TYPE_GPU;
}

hsa_status_t FirstIsa(hsa_isa_t Isa, void *Data) {
  *static_cast<hsa_isa_t *>(Data) = Isa;
  return HSA_STATUS_INFO_BREAK;
}

IsaView AgentView(hsa_agent_t Agent) {
  std::lock_guard<std::mutex> Lock(StateMutex);
  auto It = ViewsByAgent.find(Agent.handle);
  return It == ViewsByAgent.end() ? IsaView{} : It->second;
}

IsaView VirtualView(hsa_isa_t Isa) {
  std::lock_guard<std::mutex> Lock(StateMutex);
  auto It = ViewsByIsa.find(Isa.handle);
  return It == ViewsByIsa.end() ? IsaView{} : It->second;
}

hsa_status_t DiscoverAgent(hsa_agent_t Agent, void *) {
  if (!IsGpu(Agent))
    return HSA_STATUS_SUCCESS;
  hsa_isa_t Physical{};
  hsa_status_t Status = Next_hsa_agent_iterate_isas(Agent, FirstIsa, &Physical);
  if ((Status != HSA_STATUS_SUCCESS && Status != HSA_STATUS_INFO_BREAK) ||
      !Physical.handle)
    return HSA_STATUS_ERROR_INVALID_ISA;
  IsaView View;
  View.Agent = Agent;
  View.Physical = Physical;
  View.Presented.handle = VirtualIsaTag | (Agent.handle & 0xffffffffULL);
  View.Wave.handle = VirtualWaveTag | (Agent.handle & 0xffffffffULL);
  View.PhysicalName = PhysicalIsaName(Physical);
  View.PresentedName = "amdgcn-amd-amdhsa--" + PresentedGfx;
  if (View.PhysicalName.empty())
    return HSA_STATUS_ERROR_INVALID_ISA;
  std::lock_guard<std::mutex> Lock(StateMutex);
  ViewsByAgent[Agent.handle] = View;
  ViewsByIsa[View.Presented.handle] = View;
  ViewsByWave[View.Wave.handle] = View;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API ToolAgentGetInfo(hsa_agent_t Agent,
                                      hsa_agent_info_t Attribute, void *Value) {
  if (!Value)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!IsGpu(Agent))
    return Next_hsa_agent_get_info(Agent, Attribute, Value);
  IsaView View = AgentView(Agent);
  if (Attribute == HSA_AGENT_INFO_ISA) {
    *static_cast<hsa_isa_t *>(Value) = View.Presented;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_AGENT_INFO_NAME) {
    std::memset(Value, 0, 64);
    std::strncpy(static_cast<char *>(Value), PresentedGfx.c_str(), 63);
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_AGENT_INFO_WAVEFRONT_SIZE) {
    *static_cast<uint32_t *>(Value) = 32;
    return HSA_STATUS_SUCCESS;
  }
  return Next_hsa_agent_get_info(Agent, Attribute, Value);
}

hsa_status_t HSA_API ToolAgentIterateIsas(hsa_agent_t Agent,
                                          hsa_status_t (*Callback)(hsa_isa_t,
                                                                   void *),
                                          void *Data) {
  if (!Callback)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!IsGpu(Agent))
    return Next_hsa_agent_iterate_isas(Agent, Callback, Data);
  return Callback(AgentView(Agent).Presented, Data);
}

hsa_status_t HSA_API ToolIsaFromName(const char *Name, hsa_isa_t *Isa) {
  if (!Name || !Isa)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (Processor(Name) != PresentedGfx)
    return Next_hsa_isa_from_name(Name, Isa);
  std::lock_guard<std::mutex> Lock(StateMutex);
  if (ViewsByIsa.empty())
    return HSA_STATUS_ERROR_INVALID_ISA_NAME;
  *Isa = ViewsByIsa.begin()->second.Presented;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t VirtualIsaInfo(hsa_isa_t Isa, hsa_isa_info_t Attribute,
                            void *Value) {
  if (!Value)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  IsaView View = VirtualView(Isa);
  if (!View.Presented.handle)
    return HSA_STATUS_ERROR_INVALID_ISA;
  if (Attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(Value) = View.PresentedName.size() + 1;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(Value, View.PresentedName.c_str(),
                View.PresentedName.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_ISA_INFO_CALL_CONVENTION_INFO_WAVEFRONT_SIZE) {
    *static_cast<uint32_t *>(Value) = 32;
    return HSA_STATUS_SUCCESS;
  }
  return Next_hsa_isa_get_info_alt(View.Physical, Attribute, Value);
}

hsa_status_t HSA_API ToolIsaGetInfoAlt(hsa_isa_t Isa, hsa_isa_info_t Attribute,
                                       void *Value) {
  return (Isa.handle & VirtualTagMask) == VirtualIsaTag
             ? VirtualIsaInfo(Isa, Attribute, Value)
             : Next_hsa_isa_get_info_alt(Isa, Attribute, Value);
}

hsa_status_t HSA_API ToolIsaGetInfo(hsa_isa_t Isa, hsa_isa_info_t Attribute,
                                    uint32_t Index, void *Value) {
  return (Isa.handle & VirtualTagMask) == VirtualIsaTag
             ? VirtualIsaInfo(Isa, Attribute, Value)
             : Next_hsa_isa_get_info(Isa, Attribute, Index, Value);
}

hsa_status_t HSA_API ToolIsaCompatible(hsa_isa_t Code, hsa_isa_t Agent,
                                       bool *Result) {
  if (!Result)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  bool CV = (Code.handle & VirtualTagMask) == VirtualIsaTag;
  bool AV = (Agent.handle & VirtualTagMask) == VirtualIsaTag;
  if (CV || AV) {
    *Result = CV && AV;
    return HSA_STATUS_SUCCESS;
  }
  return Next_hsa_isa_compatible(Code, Agent, Result);
}

hsa_status_t HSA_API ToolExceptionPolicies(hsa_isa_t Isa, hsa_profile_t Profile,
                                           uint16_t *Mask) {
  IsaView View = VirtualView(Isa);
  return Next_hsa_isa_get_exception_policies(
      View.Presented.handle ? View.Physical : Isa, Profile, Mask);
}

hsa_status_t HSA_API ToolRoundMethod(hsa_isa_t Isa, hsa_fp_type_t Type,
                                     hsa_flush_mode_t Flush,
                                     hsa_round_method_t *Method) {
  IsaView View = VirtualView(Isa);
  return Next_hsa_isa_get_round_method(
      View.Presented.handle ? View.Physical : Isa, Type, Flush, Method);
}

hsa_status_t HSA_API ToolIterateWavefronts(
    hsa_isa_t Isa, hsa_status_t (*Callback)(hsa_wavefront_t, void *),
    void *Data) {
  if (!Callback)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  IsaView View = VirtualView(Isa);
  return View.Presented.handle
             ? Callback(View.Wave, Data)
             : Next_hsa_isa_iterate_wavefronts(Isa, Callback, Data);
}

hsa_status_t HSA_API ToolWavefrontGetInfo(hsa_wavefront_t Wave,
                                          hsa_wavefront_info_t Attribute,
                                          void *Value) {
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    if (ViewsByWave.find(Wave.handle) == ViewsByWave.end())
      return Next_hsa_wavefront_get_info(Wave, Attribute, Value);
  }
  if (!Value || Attribute != HSA_WAVEFRONT_INFO_SIZE)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *static_cast<uint32_t *>(Value) = 32;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API ToolReaderMemory(const void *Object, size_t Size,
                                      hsa_code_object_reader_t *Reader) {
  if (!Object || !Size || !Reader)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_status_t Status =
      Next_hsa_code_object_reader_create_from_memory(Object, Size, Reader);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  try {
    Bytes Copy = std::make_shared<std::vector<uint8_t>>(
        static_cast<const uint8_t *>(Object),
        static_cast<const uint8_t *>(Object) + Size);
    std::lock_guard<std::mutex> Lock(StateMutex);
    Readers[Reader->handle] = std::move(Copy);
  } catch (const std::bad_alloc &) {
    Next_hsa_code_object_reader_destroy(*Reader);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API ToolReaderFile(hsa_file_t File,
                                    hsa_code_object_reader_t *Reader) {
  io::file_pos_t Saved = 0;
  io::file_pos_t Size = 0;
  if (!io::get_file_bounds(File, &Saved, &Size) ||
      Size > static_cast<io::file_pos_t>(std::numeric_limits<size_t>::max()))
    return HSA_STATUS_ERROR_INVALID_FILE;
  io::restore_file_pos(File, 0);
  try {
    Bytes Copy =
        std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(Size));
    if (!io::read_all(File, Copy->data(), Copy->size())) {
      io::restore_file_pos(File, Saved);
      return HSA_STATUS_ERROR_INVALID_FILE;
    }
    io::restore_file_pos(File, Saved);
    hsa_status_t Status = Next_hsa_code_object_reader_create_from_memory(
        Copy->data(), Copy->size(), Reader);
    if (Status == HSA_STATUS_SUCCESS) {
      std::lock_guard<std::mutex> Lock(StateMutex);
      Readers[Reader->handle] = std::move(Copy);
    }
    return Status;
  } catch (const std::bad_alloc &) {
    io::restore_file_pos(File, Saved);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
}

hsa_status_t ToolReaderFileSlice(hsa_file_t File, size_t Offset, size_t Size,
                                 hsa_code_object_reader_t *Reader) {
  if (!Size || !Reader)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  io::file_pos_t Saved = 0;
  io::file_pos_t FileSize = 0;
  if (!io::get_file_bounds(File, &Saved, &FileSize) || Offset > FileSize ||
      Size > FileSize - Offset)
    return HSA_STATUS_ERROR_INVALID_FILE;
  io::restore_file_pos(File, Offset);
  try {
    Bytes Copy = std::make_shared<std::vector<uint8_t>>(Size);
    if (!io::read_all(File, Copy->data(), Copy->size())) {
      io::restore_file_pos(File, Saved);
      return HSA_STATUS_ERROR_INVALID_FILE;
    }
    io::restore_file_pos(File, Saved);
    hsa_status_t Status = Next_hsa_code_object_reader_create_from_memory(
        Copy->data(), Copy->size(), Reader);
    if (Status == HSA_STATUS_SUCCESS) {
      std::lock_guard<std::mutex> Lock(StateMutex);
      Readers[Reader->handle] = std::move(Copy);
    }
    return Status;
  } catch (const std::bad_alloc &) {
    io::restore_file_pos(File, Saved);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
}

void PatchLoaderTable(size_t TableLength, void *Table) {
  constexpr size_t FieldEnd =
      offsetof(
          hsa_ven_amd_loader_1_03_pfn_t,
          hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size) +
      sizeof(
          decltype(hsa_ven_amd_loader_1_03_pfn_t::
                       hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size));
  if (Table && TableLength >= FieldEnd) {
    auto *Loader = static_cast<hsa_ven_amd_loader_1_03_pfn_t *>(Table);
    Loader
        ->hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size =
        ToolReaderFileSlice;
  }
}

hsa_status_t HSA_API ToolGetExtensionTable(uint16_t Extension, uint16_t Major,
                                           uint16_t Minor, void *Table) {
  hsa_status_t Status =
      Next_hsa_system_get_extension_table(Extension, Major, Minor, Table);
  if (Status == HSA_STATUS_SUCCESS && Extension == HSA_EXTENSION_AMD_LOADER &&
      Major == 1 && Minor >= 2) {
    const size_t Length = Minor == 2 ? sizeof(hsa_ven_amd_loader_1_02_pfn_t)
                                     : sizeof(hsa_ven_amd_loader_1_03_pfn_t);
    PatchLoaderTable(Length, Table);
  }
  return Status;
}

hsa_status_t HSA_API ToolGetMajorExtensionTable(uint16_t Extension,
                                                uint16_t Major,
                                                size_t TableLength,
                                                void *Table) {
  hsa_status_t Status = Next_hsa_system_get_major_extension_table(
      Extension, Major, TableLength, Table);
  if (Status == HSA_STATUS_SUCCESS && Extension == HSA_EXTENSION_AMD_LOADER &&
      Major == 1)
    PatchLoaderTable(TableLength, Table);
  return Status;
}

hsa_status_t HSA_API ToolReaderDestroy(hsa_code_object_reader_t Reader) {
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    Readers.erase(Reader.handle);
  }
  return Next_hsa_code_object_reader_destroy(Reader);
}

Bytes ReaderData(hsa_code_object_reader_t Reader) {
  std::lock_guard<std::mutex> Lock(StateMutex);
  auto It = Readers.find(Reader.handle);
  return It == Readers.end() ? Bytes{} : It->second;
}

hsa_status_t HSA_API ToolLoad(hsa_executable_t Executable, hsa_agent_t Agent,
                              hsa_code_object_reader_t Reader,
                              const char *Options,
                              hsa_loaded_code_object_t *Loaded) {
  Bytes Source = ReaderData(Reader);
  if (!Source)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
  IsaView View = AgentView(Agent);
  if (!View.Presented.handle)
    return Next_hsa_executable_load_agent_code_object(Executable, Agent, Reader,
                                                      Options, Loaded);
  const Elf64_Ehdr *Header = GetElf(Source->data(), Source->size());
  if (!Header)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  std::string SourceGfx = GfxFromMach(Header->e_flags);
  std::string TargetGfx = Processor(View.PhysicalName);
  if (SourceGfx != PresentedGfx) {
    RejectedObjectCount.fetch_add(1, std::memory_order_relaxed);
    Proof("\"event\":\"code_object_reject\",\"source_gfx\":\"" +
          JsonEscape(SourceGfx.empty() ? "unknown" : SourceGfx) + "\"");
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }
  std::vector<std::string> Names;
  std::string Failure;
  if (!ListKernels(Source, Names, Failure)) {
    std::fprintf(stderr, "hotswap-tool: rejecting source object: %s\n",
                 Failure.c_str());
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  Bytes Skeleton = MakeSkeleton(Source, TargetGfx, Failure);
  if (!Skeleton)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  hsa_code_object_reader_t SkeletonReader{};
  hsa_status_t Status = Next_hsa_code_object_reader_create_from_memory(
      Skeleton->data(), Skeleton->size(), &SkeletonReader);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  Status = Next_hsa_executable_load_agent_code_object(
      Executable, Agent, SkeletonReader, Options, Loaded);
  Next_hsa_code_object_reader_destroy(SkeletonReader);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  auto Object = std::make_shared<LazyObject>();
  Object->Source = Source;
  Object->Agent = Agent;
  Object->Parent = Executable;
  Object->SourceGfx = SourceGfx;
  Object->TargetGfx = TargetGfx;
  Object->SourceIsa = "amdgcn-amd-amdhsa--" + SourceGfx;
  Object->TargetIsa = View.PhysicalName;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    auto &Map = ObjectsByExecutable[Executable.handle];
    for (const std::string &Name : Names) {
      if (Map.find(Name) != Map.end())
        Refuse("duplicate kernel symbol " + Name);
      Map[Name] = Object;
    }
    SkeletonStorage.push_back(std::move(Skeleton));
  }
  Proof("\"event\":\"lazy_source_registered\",\"source_gfx\":\"" + SourceGfx +
        "\",\"target_gfx\":\"" + TargetGfx +
        "\",\"kernel_count\":" + std::to_string(Names.size()));
  RegisteredObjectCount.fetch_add(1, std::memory_order_relaxed);
  return HSA_STATUS_SUCCESS;
}

void Associate(hsa_executable_t Executable, const char *Name,
               hsa_executable_symbol_t Symbol) {
  if (!Name || !Symbol.handle)
    return;
  std::string Canonical = KernelName(Name);
  std::lock_guard<std::mutex> Lock(StateMutex);
  auto E = ObjectsByExecutable.find(Executable.handle);
  if (E == ObjectsByExecutable.end())
    return;
  auto O = E->second.find(Canonical);
  if (O != E->second.end())
    Symbols[Symbol.handle] = {O->second, Canonical};
}

hsa_status_t HSA_API ToolGetSymbolByName(hsa_executable_t Executable,
                                         const char *Name,
                                         const hsa_agent_t *Agent,
                                         hsa_executable_symbol_t *Symbol) {
  hsa_status_t Status =
      Next_hsa_executable_get_symbol_by_name(Executable, Name, Agent, Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    Associate(Executable, Name, *Symbol);
  return Status;
}

hsa_status_t HSA_API ToolGetSymbol(hsa_executable_t Executable,
                                   const char *Module, const char *Name,
                                   hsa_agent_t Agent, int32_t Convention,
                                   hsa_executable_symbol_t *Symbol) {
  hsa_status_t Status = Next_hsa_executable_get_symbol(
      Executable, Module, Name, Agent, Convention, Symbol);
  if (Status == HSA_STATUS_SUCCESS)
    Associate(Executable, Name, *Symbol);
  return Status;
}

std::string GetSymbolName(hsa_executable_symbol_t Symbol) {
  uint32_t Length = 0;
  if (Next_hsa_executable_symbol_get_info(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &Length) !=
          HSA_STATUS_SUCCESS ||
      !Length)
    return {};
  std::vector<char> Name(Length + 1, 0);
  if (Next_hsa_executable_symbol_get_info(Symbol,
                                          HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                                          Name.data()) != HSA_STATUS_SUCCESS)
    return {};
  return Name.data();
}

hsa_status_t HSA_API ToolSymbolInfo(hsa_executable_symbol_t Symbol,
                                    hsa_executable_symbol_info_t Attribute,
                                    void *Value) {
  hsa_status_t Status =
      Next_hsa_executable_symbol_get_info(Symbol, Attribute, Value);
  if (Status != HSA_STATUS_SUCCESS ||
      Attribute != HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT || !Value)
    return Status;
  SymbolRecord SR;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    auto It = Symbols.find(Symbol.handle);
    if (It == Symbols.end())
      return Status;
    SR = It->second;
  }
  uint64_t Address = *static_cast<uint64_t *>(Value);
  std::shared_ptr<KernelRecord> Record;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    auto It = Kernels.find(Address);
    if (It == Kernels.end()) {
      Record = std::make_shared<KernelRecord>();
      Record->Object = SR.Object;
      Record->Name = SR.Name;
      Record->SourceObject = Address;
      Kernels[Address] = Record;
    } else {
      Record = It->second;
    }
  }
  Next_hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &Record->SourcePrivate);
  Next_hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &Record->SourceGroup);
  return Status;
}

struct IterData {
  hsa_executable_t Executable{};
  hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *);
  void *Data;
};

hsa_status_t IterCallback(hsa_executable_t Executable,
                          hsa_executable_symbol_t Symbol, void *Raw) {
  auto *D = static_cast<IterData *>(Raw);
  std::string Name = GetSymbolName(Symbol);
  Associate(D->Executable, Name.c_str(), Symbol);
  return D->Callback(Executable, Symbol, D->Data);
}

hsa_status_t HSA_API ToolIterSymbols(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *),
    void *Data) {
  IterData D{Executable, Callback, Data};
  return Next_hsa_executable_iterate_symbols(Executable, IterCallback, &D);
}

struct AgentIterData {
  hsa_executable_t Executable{};
  hsa_status_t (*Callback)(hsa_executable_t, hsa_agent_t,
                           hsa_executable_symbol_t, void *);
  void *Data;
};

hsa_status_t AgentIterCallback(hsa_executable_t Executable, hsa_agent_t Agent,
                               hsa_executable_symbol_t Symbol, void *Raw) {
  auto *D = static_cast<AgentIterData *>(Raw);
  std::string Name = GetSymbolName(Symbol);
  Associate(D->Executable, Name.c_str(), Symbol);
  return D->Callback(Executable, Agent, Symbol, D->Data);
}

hsa_status_t HSA_API
ToolIterAgentSymbols(hsa_executable_t Executable, hsa_agent_t Agent,
                     hsa_status_t (*Callback)(hsa_executable_t, hsa_agent_t,
                                              hsa_executable_symbol_t, void *),
                     void *Data) {
  AgentIterData D{Executable, Callback, Data};
  return Next_hsa_executable_iterate_agent_symbols(Executable, Agent,
                                                   AgentIterCallback, &D);
}

hsa_status_t HSA_API ToolIterProgramSymbols(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *),
    void *Data) {
  IterData D{Executable, Callback, Data};
  return Next_hsa_executable_iterate_program_symbols(Executable, IterCallback,
                                                     &D);
}

void DestroyChildren(std::vector<ChildRecord> &ChildrenToDestroy) {
  for (const ChildRecord &Child : ChildrenToDestroy) {
    if (!Child.Executable.handle)
      continue;
    if (Next_hsa_executable_destroy(Child.Executable) != HSA_STATUS_SUCCESS)
      Refuse("cannot destroy translated child executable");
  }
}

hsa_status_t HSA_API ToolExecutableDestroy(hsa_executable_t Executable) {
  std::vector<ChildRecord> ChildrenToDestroy;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    for (auto It = Children.begin(); It != Children.end();) {
      if (It->Parent.handle == Executable.handle) {
        ChildrenToDestroy.push_back(std::move(*It));
        It = Children.erase(It);
      } else {
        ++It;
      }
    }
    ObjectsByExecutable.erase(Executable.handle);
    for (auto It = Symbols.begin(); It != Symbols.end();) {
      if (It->second.Object->Parent.handle == Executable.handle)
        It = Symbols.erase(It);
      else
        ++It;
    }
    for (auto It = Kernels.begin(); It != Kernels.end();) {
      if (It->second->Object->Parent.handle == Executable.handle)
        It = Kernels.erase(It);
      else
        ++It;
    }
  }
  DestroyChildren(ChildrenToDestroy);
  return Next_hsa_executable_destroy(Executable);
}

std::string ResultString(amd_comgr_hotswap_transpile_result_t Result,
                         amd_comgr_hotswap_transpile_result_string_t Field) {
  size_t Size = 0;
  if (!Result.handle ||
      amd_comgr_hotswap_transpile_result_get_string(
          Result, Field, &Size, nullptr) != AMD_COMGR_STATUS_SUCCESS ||
      !Size)
    return {};
  std::vector<char> Text(Size + 1, 0);
  if (amd_comgr_hotswap_transpile_result_get_string(
          Result, Field, &Size, Text.data()) != AMD_COMGR_STATUS_SUCCESS)
    return {};
  return Text.data();
}

bool Translate(KernelRecord &Record) {
  amd_comgr_data_t Input{};
  amd_comgr_data_t Output{};
  amd_comgr_hotswap_transpile_result_t Result{};
  if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Input) !=
          AMD_COMGR_STATUS_SUCCESS ||
      amd_comgr_set_data(
          Input, Record.Object->Source->size(),
          reinterpret_cast<const char *>(Record.Object->Source->data())) !=
          AMD_COMGR_STATUS_SUCCESS) {
    if (Input.handle)
      amd_comgr_release_data(Input);
    Record.Failure = "cannot create COMGR input";
    return false;
  }
  amd_comgr_hotswap_transpile_options_v2_t Options{};
  Options.version = AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_VERSION_2;
  Options.cache_directory = CacheDir.empty() ? nullptr : CacheDir.c_str();
  Options.kernel_name = Record.Name.c_str();
  Options.flags = AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME |
                  AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_STRICT;
  if (AssumeHipGlobalOffsetZero)
    Options.flags |=
        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_ASSUME_HIP_GLOBAL_OFFSET_ZERO;
  amd_comgr_status_t CS = amd_comgr_hotswap_transpile_with_options_v2(
      Input, Record.Object->SourceIsa.c_str(), Record.Object->TargetIsa.c_str(),
      &Options, &Output, &Result);
  amd_comgr_release_data(Input);
  if (CS != AMD_COMGR_STATUS_SUCCESS) {
    Record.Failure =
        ResultString(Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL);
    if (Record.Failure.empty())
      Record.Failure = "COMGR per-kernel translation failed";
    if (Output.handle)
      amd_comgr_release_data(Output);
    if (Result.handle)
      amd_comgr_destroy_hotswap_transpile_result(Result);
    return false;
  }
  int64_t Scale = 1;
  amd_comgr_hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SCALED_DISPATCH_FACTOR,
      &Scale);
  size_t Size = 0;
  if (Scale < 1 || Scale > UINT32_MAX ||
      amd_comgr_get_data(Output, &Size, nullptr) != AMD_COMGR_STATUS_SUCCESS ||
      !Size) {
    Record.Failure = "invalid COMGR result";
    amd_comgr_release_data(Output);
    amd_comgr_destroy_hotswap_transpile_result(Result);
    return false;
  }
  Bytes Target;
  try {
    Target = std::make_shared<std::vector<uint8_t>>(Size);
  } catch (const std::bad_alloc &) {
    Record.Failure = "cannot allocate translated ELF storage";
    amd_comgr_release_data(Output);
    amd_comgr_destroy_hotswap_transpile_result(Result);
    return false;
  }
  if (amd_comgr_get_data(Output, &Size,
                         reinterpret_cast<char *>(Target->data())) !=
      AMD_COMGR_STATUS_SUCCESS) {
    Record.Failure = "cannot read translated ELF";
    amd_comgr_release_data(Output);
    amd_comgr_destroy_hotswap_transpile_result(Result);
    return false;
  }
  amd_comgr_release_data(Output);

  hsa_executable_t Child{};
  hsa_code_object_reader_t Reader{};
  hsa_executable_symbol_t Symbol{};
  hsa_status_t HS = Core->hsa_executable_create_alt_fn(
      HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
      &Child);
  if (HS == HSA_STATUS_SUCCESS)
    HS = Next_hsa_code_object_reader_create_from_memory(
        Target->data(), Target->size(), &Reader);
  if (HS == HSA_STATUS_SUCCESS)
    HS = Next_hsa_executable_load_agent_code_object(Child, Record.Object->Agent,
                                                    Reader, nullptr, nullptr);
  if (Reader.handle)
    Next_hsa_code_object_reader_destroy(Reader);
  if (HS == HSA_STATUS_SUCCESS)
    HS = Core->hsa_executable_freeze_fn(Child, nullptr);
  hsa_agent_t Agent = Record.Object->Agent;
  if (HS == HSA_STATUS_SUCCESS)
    HS = Next_hsa_executable_get_symbol_by_name(Child, Record.Name.c_str(),
                                                &Agent, &Symbol);
  if (HS != HSA_STATUS_SUCCESS) {
    std::string Descriptor = Record.Name + ".kd";
    HS = Next_hsa_executable_get_symbol_by_name(Child, Descriptor.c_str(),
                                                &Agent, &Symbol);
  }
  if (HS != HSA_STATUS_SUCCESS ||
      Next_hsa_executable_symbol_get_info(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
          &Record.TargetObject) != HSA_STATUS_SUCCESS ||
      Next_hsa_executable_symbol_get_info(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
          &Record.TargetPrivate) != HSA_STATUS_SUCCESS ||
      Next_hsa_executable_symbol_get_info(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
          &Record.TargetGroup) != HSA_STATUS_SUCCESS ||
      !Record.TargetObject) {
    Record.Failure = "cannot load or resolve translated target kernel";
    if (Child.handle)
      Core->hsa_executable_destroy_fn(Child);
    amd_comgr_destroy_hotswap_transpile_result(Result);
    return false;
  }
  Record.Scale = static_cast<uint32_t>(Scale);
  bool CacheHit = false;
  amd_comgr_hotswap_transpile_result_get_info(
      Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT, &CacheHit);
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    Children.push_back({Record.Object->Parent, Child, Target});
  }
  SuccessfulTranslationCount.fetch_add(1, std::memory_order_relaxed);
  Proof("\"event\":\"translation_succeeded\",\"kernel\":\"" +
        JsonEscape(Record.Name) + "\",\"source_gfx\":\"" +
        Record.Object->SourceGfx + "\",\"target_gfx\":\"" +
        Record.Object->TargetGfx +
        "\",\"cache_hit\":" + (CacheHit ? "true" : "false") +
        ",\"scale\":" + std::to_string(Record.Scale));
  amd_comgr_destroy_hotswap_transpile_result(Result);
  return true;
}

void PatchKernel(uint64_t &KernelObject, uint32_t &PrivateSegment,
                 uint32_t &GroupSegment, uint16_t &WorkgroupSizeX,
                 uint32_t *GridSizeX) {
  std::shared_ptr<KernelRecord> Record;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    auto It = Kernels.find(KernelObject);
    if (It != Kernels.end())
      Record = It->second;
  }
  if (!Record)
    Refuse("unregistered kernel_object " + std::to_string(KernelObject));
  std::lock_guard<std::mutex> Lock(Record->Mutex);
  if (!Record->Attempted) {
    Record->Attempted = true;
    Record->Succeeded = Translate(*Record);
  }
  if (!Record->Succeeded)
    Proof("\"event\":\"translation_failed\",\"kernel\":\"" +
          JsonEscape(Record->Name) + "\",\"reason\":\"" +
          JsonEscape(Record->Failure) + "\"");
  if (!Record->Succeeded)
    Refuse("translation of " + Record->Name + " failed: " + Record->Failure);
  if (PrivateSegment < Record->SourcePrivate ||
      GroupSegment < Record->SourceGroup)
    Refuse("source segment-size invariant failed for " + Record->Name);
  uint64_t Private =
      PrivateSegment - Record->SourcePrivate + Record->TargetPrivate;
  uint64_t Group = GroupSegment - Record->SourceGroup + Record->TargetGroup;
  if (Private > UINT32_MAX || Group > UINT32_MAX)
    Refuse("translated segment size overflows for " + Record->Name);
  if (Record->Scale > 1) {
    uint64_t Workgroup = static_cast<uint64_t>(WorkgroupSizeX) * Record->Scale;
    uint64_t Grid =
        GridSizeX ? static_cast<uint64_t>(*GridSizeX) * Record->Scale : 0;
    if (Workgroup > UINT16_MAX || (GridSizeX && Grid > UINT32_MAX))
      Refuse("scaled dispatch overflows for " + Record->Name);
    WorkgroupSizeX = Workgroup;
    if (GridSizeX)
      *GridSizeX = Grid;
  }
  uint64_t Source = KernelObject;
  KernelObject = Record->TargetObject;
  PrivateSegment = Private;
  GroupSegment = Group;
  RewrittenDispatchCount.fetch_add(1, std::memory_order_relaxed);
  Proof("\"event\":\"dispatch_rewritten\",\"kernel\":\"" +
        JsonEscape(Record->Name) + "\",\"source_gfx\":\"" +
        Record->Object->SourceGfx + "\",\"target_gfx\":\"" +
        Record->Object->TargetGfx +
        "\",\"source_kernel_object\":" + std::to_string(Source) +
        ",\"target_kernel_object\":" + std::to_string(KernelObject));
}

void PatchDispatch(hsa_kernel_dispatch_packet_t &Packet) {
  PatchKernel(Packet.kernel_object, Packet.private_segment_size,
              Packet.group_segment_size, Packet.workgroup_size_x,
              &Packet.grid_size_x);
}

void PatchExtendedDispatch(hsa_amd_ext_kernel_dispatch_packet_t &Packet) {
  PatchKernel(Packet.kernel_object, Packet.private_segment_size,
              Packet.group_segment_size, Packet.workgroup_size_x, nullptr);
}

void LowerExtendedDispatch(hsa_kernel_dispatch_packet_t &Storage) {
  auto &Packet =
      reinterpret_cast<hsa_amd_ext_kernel_dispatch_packet_t &>(Storage);
  PatchExtendedDispatch(Packet);
  if (Packet.reserved0 != 0)
    Refuse("extended dispatch has nonzero reserved fields");
  if (Packet.dep_signal.handle != 0)
    Refuse("extended dispatch dependency signal cannot be represented on the "
           "target");
  if (Packet.cluster_size_x != 1 || Packet.cluster_size_y != 1 ||
      Packet.cluster_size_z != 1)
    Refuse("clustered extended dispatch is unsupported on the target");
  if (!Packet.workgroup_size_x || !Packet.workgroup_size_y ||
      !Packet.workgroup_size_z || !Packet.cluster_count_x ||
      !Packet.cluster_count_y || !Packet.cluster_count_z)
    Refuse("extended dispatch has an invalid zero dimension");
  const uint64_t GridX =
      static_cast<uint64_t>(Packet.cluster_count_x) * Packet.workgroup_size_x;
  const uint64_t GridY =
      static_cast<uint64_t>(Packet.cluster_count_y) * Packet.workgroup_size_y;
  const uint64_t GridZ =
      static_cast<uint64_t>(Packet.cluster_count_z) * Packet.workgroup_size_z;
  if (GridX > UINT32_MAX || GridY > UINT32_MAX || GridZ > UINT32_MAX)
    Refuse("lowered extended dispatch grid overflows");

  hsa_kernel_dispatch_packet_t Lowered{};
  constexpr uint16_t TypeMask = ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1)
                                << HSA_PACKET_HEADER_TYPE;
  Lowered.header = (Packet.header & ~TypeMask) |
                   (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
  Lowered.setup = Packet.setup;
  Lowered.workgroup_size_x = Packet.workgroup_size_x;
  Lowered.workgroup_size_y = Packet.workgroup_size_y;
  Lowered.workgroup_size_z = Packet.workgroup_size_z;
  Lowered.grid_size_x = static_cast<uint32_t>(GridX);
  Lowered.grid_size_y = static_cast<uint32_t>(GridY);
  Lowered.grid_size_z = static_cast<uint32_t>(GridZ);
  Lowered.private_segment_size = Packet.private_segment_size;
  Lowered.group_segment_size = Packet.group_segment_size;
  Lowered.kernel_object = Packet.kernel_object;
  Lowered.kernarg_address = Packet.kernarg_address;
  Lowered.completion_signal = Packet.completion_signal;
  Storage = Lowered;
  Proof("\"event\":\"extended_dispatch_lowered\"");
}

void Interceptor(const void *Packets, uint64_t Count, uint64_t, void *,
                 hsa_amd_queue_intercept_packet_writer Writer) {
  if (!Packets || !Writer)
    Refuse("invalid intercept callback arguments");
  std::vector<hsa_kernel_dispatch_packet_t> Copy(
      static_cast<const hsa_kernel_dispatch_packet_t *>(Packets),
      static_cast<const hsa_kernel_dispatch_packet_t *>(Packets) + Count);
  for (hsa_kernel_dispatch_packet_t &Packet : Copy) {
    uint16_t Type = (Packet.header >> HSA_PACKET_HEADER_TYPE) &
                    ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
    if (Type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      PatchDispatch(Packet);
    } else if (Type == HSA_PACKET_TYPE_BARRIER_AND ||
               Type == HSA_PACKET_TYPE_BARRIER_OR) {
      continue;
    } else if (Type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      auto *Vendor =
          reinterpret_cast<hsa_amd_vendor_packet_header_t *>(&Packet);
      if (Vendor->AmdFormat == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
        LowerExtendedDispatch(Packet);
      } else if (Vendor->AmdFormat == HSA_AMD_PACKET_TYPE_BARRIER_VALUE ||
                 Vendor->AmdFormat == AMD_AQL_FORMAT_INTERCEPT_MARKER) {
        continue;
      } else {
        Refuse("unsupported vendor/extended packet format " +
               std::to_string(Vendor->AmdFormat));
      }
    } else {
      Refuse("unsupported AQL packet type " + std::to_string(Type));
    }
  }
  Writer(Copy.data(), Count);
}

hsa_status_t Protect(hsa_queue_t *Queue) {
  hsa_status_t Status = NextInterceptRegister(Queue, Interceptor, nullptr);
  if (Status == HSA_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> Lock(StateMutex);
    ProtectedQueues.insert(Queue);
    ProtectedQueueCount.fetch_add(1, std::memory_order_relaxed);
    Proof("\"event\":\"queue_proxy_created\",\"queue\":" +
          std::to_string(reinterpret_cast<uintptr_t>(Queue)));
  }
  return Status;
}

hsa_status_t
CreateProtectedQueue(hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
                     void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                     void *Data, uint32_t Private, uint32_t Group,
                     hsa_queue_t **Queue) {
  if (Type != HSA_QUEUE_TYPE_MULTI)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  hsa_status_t Status = NextInterceptCreate(Agent, Size, Type, Callback, Data,
                                            Private, Group, Queue);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  Status = Protect(*Queue);
  if (Status != HSA_STATUS_SUCCESS) {
    Next_hsa_queue_destroy(*Queue);
    *Queue = nullptr;
  }
  return Status;
}

hsa_status_t HSA_API ToolQueueCreate(
    hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
    void (*Callback)(hsa_status_t, hsa_queue_t *, void *), void *Data,
    uint32_t Private, uint32_t Group, hsa_queue_t **Queue) {
  if (!IsGpu(Agent))
    return Next_hsa_queue_create(Agent, Size, Type, Callback, Data, Private,
                                 Group, Queue);
  return CreateProtectedQueue(Agent, Size, Type, Callback, Data, Private, Group,
                              Queue);
}

hsa_status_t
ToolInterceptCreate(hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
                    void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                    void *Data, uint32_t Private, uint32_t Group,
                    hsa_queue_t **Queue) {
  if (!IsGpu(Agent))
    return NextInterceptCreate(Agent, Size, Type, Callback, Data, Private,
                               Group, Queue);
  return CreateProtectedQueue(Agent, Size, Type, Callback, Data, Private, Group,
                              Queue);
}

hsa_status_t ToolAmdQueueCreate(hsa_agent_t Agent,
                                hsa_amd_queue_create_desc_t *Descs,
                                uint32_t Count) {
  if (!IsGpu(Agent))
    return NextAmdQueueCreate(Agent, Descs, Count);
  if (!Descs || !Count)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t FirstError = HSA_STATUS_SUCCESS;
  for (uint32_t I = 0; I < Count; ++I) {
    hsa_amd_queue_create_desc_t &Desc = Descs[I];
    Desc.queue = nullptr;
    if (Desc.version != HSA_AMD_QUEUE_CREATE_DESC_VERSION ||
        Desc.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE ||
        Desc.queue_size_bytes == 0 ||
        Desc.queue_size_bytes % sizeof(hsa_kernel_dispatch_packet_t) != 0 ||
        Desc.engine.compute.type != HSA_QUEUE_TYPE_MULTI ||
        Desc.traffic_class != 0 ||
        (Desc.flags & ~HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF) != 0) {
      if (FirstError == HSA_STATUS_SUCCESS)
        FirstError = HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
      continue;
    }

    uint32_t QueueSize =
        Desc.queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t);
    hsa_status_t Status = CreateProtectedQueue(
        Agent, QueueSize, Desc.engine.compute.type, Desc.callback,
        Desc.callback_data, Desc.engine.compute.private_segment_size,
        UINT32_MAX, &Desc.queue);
    if (Status == HSA_STATUS_SUCCESS &&
        Desc.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL)
      Status = NextQueueSetPriority(Desc.queue, Desc.priority);
    if (Status == HSA_STATUS_SUCCESS && Desc.engine.compute.cu_mask_count)
      Status = NextQueueSetCuMask(Desc.queue, Desc.engine.compute.cu_mask_count,
                                  Desc.engine.compute.cu_mask);
    if (Status != HSA_STATUS_SUCCESS && Desc.queue) {
      {
        std::lock_guard<std::mutex> Lock(StateMutex);
        ProtectedQueues.erase(Desc.queue);
      }
      Next_hsa_queue_destroy(Desc.queue);
      Desc.queue = nullptr;
    }
    if (Status != HSA_STATUS_SUCCESS && FirstError == HSA_STATUS_SUCCESS)
      FirstError = Status;
    if (Status == HSA_STATUS_SUCCESS &&
        (Desc.flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF))
      Proof("\"event\":\"queue_device_ring_requested\",\"queue\":" +
            std::to_string(reinterpret_cast<uintptr_t>(Desc.queue)));
  }
  return FirstError;
}

hsa_status_t HSA_API ToolQueueDestroy(hsa_queue_t *Queue) {
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    ProtectedQueues.erase(Queue);
  }
  return Next_hsa_queue_destroy(Queue);
}

hsa_status_t ToolQueueInfo(hsa_queue_t *Queue,
                           hsa_queue_info_attribute_t Attribute, void *Value) {
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    if (Attribute == HSA_AMD_QUEUE_INFO_DOORBELL_ID &&
        ProtectedQueues.count(Queue))
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return NextQueueGetInfo(Queue, Attribute, Value);
}

bool Flag(const char *Name) {
  const char *Value = std::getenv(Name);
  return Value && Value[0] && std::strcmp(Value, "0") != 0;
}

bool Configure() {
  const char *Present = std::getenv("HSA_HOTSWAP_PRESENT_ISA");
  if (!Present || !Present[0])
    return false;
  PresentedGfx = Processor(Present);
  if (PresentedGfx.empty())
    PresentedGfx = Present;
  if (PresentedGfx != "gfx1250") {
    std::fprintf(
        stderr, "hotswap-tool: prototype supports gfx1250 presentation only\n");
    return false;
  }
  if (const char *Value = std::getenv("HSA_HOTSWAP_CACHE_DIR"))
    CacheDir = Value;
  if (const char *Value = std::getenv("HSA_HOTSWAP_PROOF_LOG"))
    ProofPath = Value;
  AssumeHipGlobalOffsetZero = Flag("HSA_HOTSWAP_ASSUME_GLOBAL_OFFSET_ZERO") ||
                              Flag("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO");
  RegisteredObjectCount.store(0, std::memory_order_relaxed);
  SuccessfulTranslationCount.store(0, std::memory_order_relaxed);
  RewrittenDispatchCount.store(0, std::memory_order_relaxed);
  ProtectedQueueCount.store(0, std::memory_order_relaxed);
  RejectedObjectCount.store(0, std::memory_order_relaxed);
  return true;
}

bool Save(HsaApiTable *Table) {
  if (!Table || !Table->core_ || !Table->amd_ext_)
    return false;
  Core = Table->core_;
  Amd = Table->amd_ext_;
#define GET_CORE(Name) Next_##Name = Core->Name##_fn
  GET_CORE(hsa_iterate_agents);
  GET_CORE(hsa_agent_get_info);
  GET_CORE(hsa_isa_from_name);
  GET_CORE(hsa_agent_iterate_isas);
  GET_CORE(hsa_isa_get_info);
  GET_CORE(hsa_isa_get_info_alt);
  GET_CORE(hsa_isa_compatible);
  GET_CORE(hsa_isa_get_exception_policies);
  GET_CORE(hsa_isa_get_round_method);
  GET_CORE(hsa_isa_iterate_wavefronts);
  GET_CORE(hsa_wavefront_get_info);
  GET_CORE(hsa_system_get_extension_table);
  GET_CORE(hsa_system_get_major_extension_table);
  GET_CORE(hsa_queue_create);
  GET_CORE(hsa_queue_destroy);
  GET_CORE(hsa_code_object_reader_create_from_memory);
  GET_CORE(hsa_code_object_reader_create_from_file);
  GET_CORE(hsa_code_object_reader_destroy);
  GET_CORE(hsa_executable_destroy);
  GET_CORE(hsa_executable_load_agent_code_object);
  GET_CORE(hsa_executable_get_symbol_by_name);
  GET_CORE(hsa_executable_get_symbol);
  GET_CORE(hsa_executable_symbol_get_info);
  GET_CORE(hsa_executable_iterate_symbols);
  GET_CORE(hsa_executable_iterate_agent_symbols);
  GET_CORE(hsa_executable_iterate_program_symbols);
#undef GET_CORE
  NextInterceptCreate = Amd->hsa_amd_queue_intercept_create_fn;
  NextInterceptRegister = Amd->hsa_amd_queue_intercept_register_fn;
  NextQueueGetInfo = Amd->hsa_amd_queue_get_info_fn;
  NextAmdQueueCreate = Amd->hsa_amd_queue_create_fn;
  NextQueueSetPriority = Amd->hsa_amd_queue_set_priority_fn;
  NextQueueSetCuMask = Amd->hsa_amd_queue_cu_set_mask_fn;
  return Next_hsa_iterate_agents && Next_hsa_agent_get_info &&
         Next_hsa_isa_from_name && Next_hsa_agent_iterate_isas &&
         Next_hsa_isa_get_info && Next_hsa_isa_get_info_alt &&
         Next_hsa_isa_compatible && Next_hsa_isa_get_exception_policies &&
         Next_hsa_isa_get_round_method && Next_hsa_isa_iterate_wavefronts &&
         Next_hsa_wavefront_get_info && Next_hsa_system_get_extension_table &&
         Next_hsa_system_get_major_extension_table && Next_hsa_queue_create &&
         Next_hsa_queue_destroy &&
         Next_hsa_code_object_reader_create_from_memory &&
         Next_hsa_code_object_reader_create_from_file &&
         Next_hsa_code_object_reader_destroy && Next_hsa_executable_destroy &&
         Next_hsa_executable_load_agent_code_object &&
         Next_hsa_executable_get_symbol_by_name &&
         Next_hsa_executable_get_symbol &&
         Next_hsa_executable_symbol_get_info &&
         Next_hsa_executable_iterate_symbols &&
         Next_hsa_executable_iterate_agent_symbols &&
         Next_hsa_executable_iterate_program_symbols && NextInterceptCreate &&
         NextInterceptRegister && NextQueueGetInfo && NextAmdQueueCreate &&
         NextQueueSetPriority && NextQueueSetCuMask;
}

void Install() {
  Core->hsa_agent_get_info_fn = ToolAgentGetInfo;
  Core->hsa_isa_from_name_fn = ToolIsaFromName;
  Core->hsa_agent_iterate_isas_fn = ToolAgentIterateIsas;
  Core->hsa_isa_get_info_fn = ToolIsaGetInfo;
  Core->hsa_isa_get_info_alt_fn = ToolIsaGetInfoAlt;
  Core->hsa_isa_compatible_fn = ToolIsaCompatible;
  Core->hsa_isa_get_exception_policies_fn = ToolExceptionPolicies;
  Core->hsa_isa_get_round_method_fn = ToolRoundMethod;
  Core->hsa_isa_iterate_wavefronts_fn = ToolIterateWavefronts;
  Core->hsa_wavefront_get_info_fn = ToolWavefrontGetInfo;
  Core->hsa_system_get_extension_table_fn = ToolGetExtensionTable;
  Core->hsa_system_get_major_extension_table_fn = ToolGetMajorExtensionTable;
  Core->hsa_queue_create_fn = ToolQueueCreate;
  Core->hsa_queue_destroy_fn = ToolQueueDestroy;
  Core->hsa_code_object_reader_create_from_memory_fn = ToolReaderMemory;
  Core->hsa_code_object_reader_create_from_file_fn = ToolReaderFile;
  Core->hsa_code_object_reader_destroy_fn = ToolReaderDestroy;
  Core->hsa_executable_destroy_fn = ToolExecutableDestroy;
  Core->hsa_executable_load_agent_code_object_fn = ToolLoad;
  Core->hsa_executable_get_symbol_by_name_fn = ToolGetSymbolByName;
  Core->hsa_executable_get_symbol_fn = ToolGetSymbol;
  Core->hsa_executable_symbol_get_info_fn = ToolSymbolInfo;
  Core->hsa_executable_iterate_symbols_fn = ToolIterSymbols;
  Core->hsa_executable_iterate_agent_symbols_fn = ToolIterAgentSymbols;
  Core->hsa_executable_iterate_program_symbols_fn = ToolIterProgramSymbols;
  Amd->hsa_amd_queue_intercept_create_fn = ToolInterceptCreate;
  Amd->hsa_amd_queue_get_info_fn = ToolQueueInfo;
  Amd->hsa_amd_queue_create_fn = ToolAmdQueueCreate;
}

void Restore() {
  if (!Core || !Amd)
    return;
#define RESTORE_CORE(Name) Core->Name##_fn = Next_##Name
  RESTORE_CORE(hsa_agent_get_info);
  RESTORE_CORE(hsa_isa_from_name);
  RESTORE_CORE(hsa_agent_iterate_isas);
  RESTORE_CORE(hsa_isa_get_info);
  RESTORE_CORE(hsa_isa_get_info_alt);
  RESTORE_CORE(hsa_isa_compatible);
  RESTORE_CORE(hsa_isa_get_exception_policies);
  RESTORE_CORE(hsa_isa_get_round_method);
  RESTORE_CORE(hsa_isa_iterate_wavefronts);
  RESTORE_CORE(hsa_wavefront_get_info);
  RESTORE_CORE(hsa_system_get_extension_table);
  RESTORE_CORE(hsa_system_get_major_extension_table);
  RESTORE_CORE(hsa_queue_create);
  RESTORE_CORE(hsa_queue_destroy);
  RESTORE_CORE(hsa_code_object_reader_create_from_memory);
  RESTORE_CORE(hsa_code_object_reader_create_from_file);
  RESTORE_CORE(hsa_code_object_reader_destroy);
  RESTORE_CORE(hsa_executable_destroy);
  RESTORE_CORE(hsa_executable_load_agent_code_object);
  RESTORE_CORE(hsa_executable_get_symbol_by_name);
  RESTORE_CORE(hsa_executable_get_symbol);
  RESTORE_CORE(hsa_executable_symbol_get_info);
  RESTORE_CORE(hsa_executable_iterate_symbols);
  RESTORE_CORE(hsa_executable_iterate_agent_symbols);
  RESTORE_CORE(hsa_executable_iterate_program_symbols);
#undef RESTORE_CORE
  Amd->hsa_amd_queue_intercept_create_fn = NextInterceptCreate;
  Amd->hsa_amd_queue_get_info_fn = NextQueueGetInfo;
  Amd->hsa_amd_queue_create_fn = NextAmdQueueCreate;
}

} // namespace

extern "C" {

HSA_HOTSWAP_EXPORT bool OnLoad(HsaApiTable *Table, uint64_t, uint64_t,
                               const char *const *) {
  if (!Configure() || !Save(Table))
    return false;
  if (Next_hsa_iterate_agents(DiscoverAgent, nullptr) != HSA_STATUS_SUCCESS ||
      ViewsByAgent.empty())
    return false;
  Install();
  Proof("\"event\":\"tool_loaded\",\"presented_gfx\":\"" + PresentedGfx + "\"");
  std::fprintf(stderr, "hotswap-tool: presenting %s on protected lazy queues\n",
               PresentedGfx.c_str());
  return true;
}

HSA_HOTSWAP_EXPORT void OnUnload() {
  Restore();
  Proof("\"event\":\"coverage_summary\",\"registered_source_objects\":" +
        std::to_string(RegisteredObjectCount.load(std::memory_order_relaxed)) +
        ",\"successful_translations\":" +
        std::to_string(
            SuccessfulTranslationCount.load(std::memory_order_relaxed)) +
        ",\"rewritten_dispatches\":" +
        std::to_string(RewrittenDispatchCount.load(std::memory_order_relaxed)) +
        ",\"protected_queues_created\":" +
        std::to_string(ProtectedQueueCount.load(std::memory_order_relaxed)) +
        ",\"rejected_non_source_objects\":" +
        std::to_string(RejectedObjectCount.load(std::memory_order_relaxed)) +
        ",\"all_intercepted_dispatches_rewritten\":true");
  Proof("\"event\":\"tool_unloaded\"");
  std::vector<ChildRecord> LocalChildren;
  {
    std::lock_guard<std::mutex> Lock(StateMutex);
    if (!ProtectedQueues.empty()) {
      std::fprintf(stderr,
                   "hotswap-tool: unloading with %zu protected queues alive\n",
                   ProtectedQueues.size());
    }
    LocalChildren.swap(Children);
    ProtectedQueues.clear();
    Kernels.clear();
    Symbols.clear();
    ObjectsByExecutable.clear();
    Readers.clear();
    ViewsByAgent.clear();
    ViewsByIsa.clear();
    ViewsByWave.clear();
    SkeletonStorage.clear();
  }
  if (!LocalChildren.empty())
    Proof("\"event\":\"translated_children_deferred_to_loader\",\"count\":" +
          std::to_string(LocalChildren.size()));
  PresentedGfx.clear();
  CacheDir.clear();
  ProofPath.clear();
  AssumeHipGlobalOffsetZero = false;
  std::fprintf(stderr, "hotswap-tool: unloaded\n");
}

} // extern "C"
