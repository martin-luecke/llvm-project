//===- hotswap-tool.cpp - COMGR-backed HSA presentation tool -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This DSO owns the runtime-facing half of HotSwap. It presents a logical ISA,
// registers source code objects without handing incompatible machine code to
// the loader, translates a kernel through the public COMGR API when it is first
// dispatched, and replaces a non-executable token in every intercepted
// dispatch with the translated physical-ISA kernel object.
//
// The source object is deliberately never target-tagged or loaded. Such an
// object would leave source instructions reachable under a target ELF tag and
// would rely on generation-specific descriptors being loader-compatible.
//
//===----------------------------------------------------------------------===//

#include "hotswap-dispatch.h"
#include "hotswap-object.h"
#include "hotswap-platform-io.h"
#include "hotswap-proof.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <amd_comgr.h>
#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ext_amd.h>
#include <hsa_ven_amd_loader.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define HSA_HOTSWAP_EXPORT __attribute__((visibility("default")))

namespace COMGR::hotswap::hsa_tool {
namespace {

struct KernelRecord;
bool ensureTranslated(const std::shared_ptr<KernelRecord> &Kernel);

hsa_status_t HSA_API toolReaderMemory(const void *Object, size_t Size,
                                      hsa_code_object_reader_t *Reader);
hsa_status_t HSA_API toolReaderFile(hsa_file_t File,
                                    hsa_code_object_reader_t *Reader);
hsa_status_t HSA_API toolReaderFileSlice(hsa_file_t File, size_t Offset,
                                         size_t Size,
                                         hsa_code_object_reader_t *Reader);
hsa_status_t HSA_API toolReaderDestroy(hsa_code_object_reader_t Reader);
hsa_status_t HSA_API toolSoftQueueCreate(hsa_region_t Region, uint32_t Size,
                                         hsa_queue_type32_t Type,
                                         uint32_t Features,
                                         hsa_signal_t DoorbellSignal,
                                         hsa_queue_t **Queue);
hsa_status_t HSA_API toolLoadAgent(hsa_executable_t Executable,
                                   hsa_agent_t Agent,
                                   hsa_code_object_reader_t Reader,
                                   const char *Options,
                                   hsa_loaded_code_object_t *Loaded);
hsa_status_t HSA_API toolLoadProgram(hsa_executable_t Executable,
                                     hsa_code_object_reader_t Reader,
                                     const char *Options,
                                     hsa_loaded_code_object_t *Loaded);
hsa_status_t HSA_API toolExecutableFreeze(hsa_executable_t Executable,
                                          const char *Options);
hsa_status_t HSA_API toolExecutableValidate(hsa_executable_t Executable,
                                            uint32_t *Result);
hsa_status_t HSA_API toolExecutableValidateAlt(hsa_executable_t Executable,
                                               const char *Options,
                                               uint32_t *Result);
hsa_status_t HSA_API toolGetSymbolByName(hsa_executable_t Executable,
                                         const char *Name,
                                         const hsa_agent_t *Agent,
                                         hsa_executable_symbol_t *Symbol);
hsa_status_t HSA_API toolGetSymbol(hsa_executable_t Executable,
                                   const char *ModuleName,
                                   const char *SymbolName, hsa_agent_t Agent,
                                   int32_t CallConvention,
                                   hsa_executable_symbol_t *Symbol);
hsa_status_t HSA_API toolSymbolGetInfo(hsa_executable_symbol_t Symbol,
                                       hsa_executable_symbol_info_t Attribute,
                                       void *Value);
hsa_status_t HSA_API toolIterateSymbols(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *),
    void *Data);
hsa_status_t HSA_API toolIterateAgentSymbols(
    hsa_executable_t Executable, hsa_agent_t Agent,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_agent_t,
                             hsa_executable_symbol_t, void *),
    void *Data);
hsa_status_t HSA_API toolIterateProgramSymbols(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *),
    void *Data);
hsa_status_t HSA_API toolExecutableDestroy(hsa_executable_t Executable);
void destroyKernelChild(KernelRecord &Kernel);

static_assert(sizeof(hsa_kernel_dispatch_packet_t) == 64,
              "the interceptor requires 64-byte AQL packets");
static_assert(sizeof(hsa_amd_ext_kernel_dispatch_packet_t) == 64,
              "the interceptor requires 64-byte extended AQL packets");

struct AgentView {
  hsa_agent_t Agent{};
  hsa_isa_t PresentedIsa{};
  hsa_isa_t ExecutionIsa{};
  std::string PresentedName;
  std::string ExecutionName;
  std::string PresentedGfx;
  std::string ExecutionGfx;
  uint32_t WavefrontSize = 0;
  std::array<uint16_t, 3> MaxWorkgroupDim{};
  uint32_t MaxWorkgroupSize = 0;
  hsa_dim3_t MaxGridDim{};
  uint64_t MaxGridSize = 0;
  bool NeedsTranslation = false;
};

struct SourceObject;

// The address of this allocation is exposed as kernel_object. It is never a
// loader allocation and cannot name source machine code. Queue interception is
// the only operation that turns it into an executable address.
struct alignas(64) VirtualKernelToken {
  uint64_t Magic = 0x48534b45524e454cULL; // "HSKERNEL"
};

// HSA symbol handles are opaque. A source kernel is represented by this token
// until an operation genuinely needs its translated child executable.
struct alignas(64) VirtualSymbolToken {
  uint64_t Magic = 0x485353594d424f4cULL; // "HSSYMBOL"
};

struct KernelRecord {
  std::mutex Mutex;
  std::weak_ptr<SourceObject> Object;
  std::string MetadataName;
  std::string SymbolName;
  uint32_t KernargSegmentSize = 0;
  uint32_t KernargSegmentAlignment = 0;
  uint32_t SourcePrivateSegmentSize = 0;
  uint32_t SourceGroupSegmentSize = 0;
  uint32_t SourceWavefrontSize = 0;
  bool SourceDynamicCallstack = false;
  bool Attempted = false;
  bool Succeeded = false;
  std::string Failure;
  hsa_executable_t Child{};
  hsa_executable_symbol_t Symbol{};
  std::unique_ptr<VirtualSymbolToken> SymbolToken;
  std::unique_ptr<VirtualKernelToken> KernelToken;
  KernelDispatchTarget Target;
};

struct SourceObject {
  Bytes SourceElf;
  hsa_agent_t Agent{};
  hsa_executable_t Parent{};
  hsa_profile_t Profile = HSA_PROFILE_FULL;
  hsa_default_float_rounding_mode_t Rounding =
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT;
  std::string Options;
  std::string SourceIsa;
  std::string TargetIsa;
  std::string SourceGfx;
  std::string TargetGfx;
  std::array<uint16_t, 3> TargetMaxWorkgroupDim{};
  uint32_t TargetMaxWorkgroupSize = 0;
  hsa_dim3_t TargetMaxGridDim{};
  uint64_t TargetMaxGridSize = 0;
  std::atomic<bool> Alive{true};
  std::mutex LifetimeMutex;
  size_t ActiveIterations = 0;
  bool RetirementRequested = false;
  bool CleanupStarted = false;
  std::vector<std::shared_ptr<KernelRecord>> Kernels;
};

struct ExecutableRecord {
  std::shared_ptr<SourceObject> Object;
  bool Frozen = false;
};

struct NativeExecutableRecord {
  size_t ActiveMutations = 0;
  bool HasContent = false;
  bool SourceRegistrationActive = false;
  bool FreezeActive = false;
  bool Destroying = false;
};

struct Counters {
  std::atomic<uint64_t> RegisteredObjects{0};
  std::atomic<uint64_t> RegisteredKernels{0};
  std::atomic<uint64_t> TranslationRequests{0};
  std::atomic<uint64_t> SuccessfulTranslations{0};
  std::atomic<uint64_t> FailedTranslations{0};
  std::atomic<uint64_t> CacheHits{0};
  std::atomic<uint64_t> InterceptedDispatches{0};
  std::atomic<uint64_t> RewrittenDispatches{0};
  std::atomic<uint64_t> ProtectedQueues{0};
  std::atomic<uint64_t> RejectedObjects{0};
};

struct ProtectedQueueRecord {
  const hsa_queue_t *Queue = nullptr;
  uint64_t Capacity = 0;
};

struct ToolState {
  std::mutex Mutex;
  std::mutex ProofMutex;
  std::mutex LoaderTableMutex;
  // Packet callbacks hold a shared lock through the packet writer. Parent
  // destruction takes the exclusive lock before destroying translated child
  // executables, so a rewritten target cannot disappear between validation
  // and submission.
  std::shared_mutex DispatchMutex;
  CoreApiTable *Core = nullptr;
  AmdExtTable *Amd = nullptr;
  bool Installed = false;
  bool Active = false;
  bool ApiRestoreDeferred = false;

  std::string PresentedName;
  std::string PresentedGfx;
  hsa_isa_t PresentedIsa{};
  std::string CacheDirectory;
  std::string ProofPath;
  bool AssumeHipGlobalOffsetZero = false;

  llvm::DenseMap<uint64_t, AgentView> Agents;
  llvm::DenseMap<uint64_t, Bytes> Readers;
  llvm::DenseMap<uint64_t, ExecutableRecord> Executables;
  llvm::DenseMap<uint64_t, NativeExecutableRecord> NativeExecutables;
  // A null value denotes a retired virtual symbol. Retaining the key and its
  // allocation prevents a stale opaque handle from reaching ROCr or aliasing
  // a later executable in this process.
  llvm::DenseMap<uint64_t, std::shared_ptr<KernelRecord>> SymbolTokens;
  llvm::DenseMap<uint64_t, std::shared_ptr<KernelRecord>> KernelTokens;
  llvm::DenseMap<const hsa_queue_t *, std::unique_ptr<ProtectedQueueRecord>>
      ProtectedQueues;
  // A kernel object is presented as the address of its token. Keep retired
  // allocations until unload so an allocator cannot give a stale address to
  // an unrelated kernel later in the same process.
  std::vector<std::unique_ptr<VirtualSymbolToken>> RetiredSymbolTokens;
  std::vector<std::unique_ptr<VirtualKernelToken>> RetiredKernelTokens;
  Counters Count;

  decltype(hsa_iterate_agents) *NextIterateAgents = nullptr;
  decltype(hsa_agent_get_info) *NextAgentGetInfo = nullptr;
  decltype(hsa_isa_from_name) *NextIsaFromName = nullptr;
  decltype(hsa_isa_compatible) *NextIsaCompatible = nullptr;
  decltype(hsa_agent_iterate_isas) *NextAgentIterateIsas = nullptr;
  decltype(hsa_isa_get_info_alt) *NextIsaGetInfoAlt = nullptr;
  decltype(hsa_isa_iterate_wavefronts) *NextIsaIterateWavefronts = nullptr;
  decltype(hsa_wavefront_get_info) *NextWavefrontGetInfo = nullptr;
  decltype(hsa_system_get_extension_table) *NextGetExtensionTable = nullptr;
  decltype(hsa_system_get_major_extension_table) *NextGetMajorExtensionTable =
      nullptr;
  decltype(hsa_queue_create) *NextQueueCreate = nullptr;
  decltype(hsa_soft_queue_create) *NextSoftQueueCreate = nullptr;
  decltype(hsa_queue_destroy) *NextQueueDestroy = nullptr;
  decltype(hsa_code_object_reader_create_from_memory) *NextReaderMemory =
      nullptr;
  decltype(hsa_code_object_reader_create_from_file) *NextReaderFile = nullptr;
  decltype(hsa_code_object_reader_destroy) *NextReaderDestroy = nullptr;
  decltype(hsa_executable_create_alt) *NextExecutableCreate = nullptr;
  decltype(hsa_executable_destroy) *NextExecutableDestroy = nullptr;
  decltype(hsa_executable_freeze) *NextExecutableFreeze = nullptr;
  decltype(hsa_executable_get_info) *NextExecutableGetInfo = nullptr;
  decltype(hsa_executable_validate) *NextExecutableValidate = nullptr;
  decltype(hsa_executable_validate_alt) *NextExecutableValidateAlt = nullptr;
  decltype(hsa_executable_load_program_code_object) *NextLoadProgram = nullptr;
  decltype(hsa_executable_load_agent_code_object) *NextLoadAgent = nullptr;
  decltype(hsa_executable_load_code_object) *NextLoadCodeObject = nullptr;
  decltype(hsa_executable_global_variable_define) *NextDefineGlobal = nullptr;
  decltype(hsa_executable_agent_global_variable_define) *NextDefineAgentGlobal =
      nullptr;
  decltype(hsa_executable_readonly_variable_define) *NextDefineReadonly =
      nullptr;
  decltype(hsa_executable_get_symbol_by_name) *NextGetSymbolByName = nullptr;
  decltype(hsa_executable_get_symbol) *NextGetSymbol = nullptr;
  decltype(hsa_executable_symbol_get_info) *NextSymbolGetInfo = nullptr;
  decltype(hsa_executable_iterate_symbols) *NextIterateSymbols = nullptr;
  decltype(hsa_executable_iterate_agent_symbols) *NextIterateAgentSymbols =
      nullptr;
  decltype(hsa_executable_iterate_program_symbols) *NextIterateProgramSymbols =
      nullptr;

  decltype(hsa_amd_queue_intercept_create) *NextInterceptCreate = nullptr;
  decltype(hsa_amd_queue_intercept_register) *NextInterceptRegister = nullptr;
  decltype(hsa_amd_queue_get_info) *NextQueueGetInfo = nullptr;
  decltype(hsa_amd_queue_create) *NextAmdQueueCreate = nullptr;
  decltype(hsa_amd_queue_set_priority) *NextQueueSetPriority = nullptr;
  decltype(hsa_amd_queue_cu_set_mask) *NextQueueSetCuMask = nullptr;

  decltype(hsa_ven_amd_loader_query_host_address) *NextLoaderHostAddress =
      nullptr;
  decltype(hsa_ven_amd_loader_query_segment_descriptors)
      *NextLoaderQuerySegments = nullptr;
  decltype(hsa_ven_amd_loader_query_executable) *NextLoaderExecutable = nullptr;
  decltype(hsa_ven_amd_loader_executable_iterate_loaded_code_objects)
      *NextLoaderIterateLoaded = nullptr;
  decltype(hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size)
      *NextLoaderReaderFileSlice = nullptr;
  decltype(hsa_ven_amd_loader_iterate_executables)
      *NextLoaderIterateExecutables = nullptr;
};

constexpr size_t AgentNameSize = 64;

// The HSA tool ABI has process-wide callbacks and function tables. This is the
// sole mutable global; it is allocated by OnLoad and deleted by OnUnload, so it
// has no static constructor and its lifetime matches the runtime tool contract.
ToolState *State = nullptr;

[[noreturn]] void refuse(const std::string &Reason) {
  llvm::errs() << "hotswap-hsa-tool: refusing execution: " << Reason << '\n';
  llvm::errs().flush();
  std::abort();
}

bool writeProof(const std::string &Fields) {
  if (!State || State->ProofPath.empty())
    return true;
  std::lock_guard<std::mutex> Lock(State->ProofMutex);
  if (!appendProofLine({State->ProofPath, Fields, processId()})) {
    llvm::errs() << "hotswap-hsa-tool: cannot open proof log "
                 << State->ProofPath << '\n';
    return false;
  }
  return true;
}

void proofOrRefuse(const std::string &Fields) {
  if (!writeProof(Fields))
    refuse("configured proof log could not be written");
}

std::string processor(const std::string &IsaName) {
  const size_t Begin = IsaName.find("gfx");
  if (Begin == std::string::npos)
    return {};
  size_t End = Begin + 3;
  while (End != IsaName.size()) {
    const char C = IsaName[End];
    if (!((C >= '0' && C <= '9') || (C >= 'a' && C <= 'z') ||
          (C >= 'A' && C <= 'Z')))
      break;
    ++End;
  }
  return IsaName.substr(Begin, End - Begin);
}

std::string isaName(hsa_isa_t Isa) {
  uint32_t Size = 0;
  if (!State ||
      State->NextIsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME_LENGTH, &Size) !=
          HSA_STATUS_SUCCESS ||
      Size == 0 ||
      static_cast<size_t>(Size) == std::numeric_limits<size_t>::max())
    return {};
  // The HSA specification excludes the terminator from NAME_LENGTH, while
  // current ROCr includes it. Keep one extra zero byte and accept only these
  // two exact layouts; an earlier terminator still denotes malformed data.
  std::unique_ptr<char[]> Name(
      new (std::nothrow) char[static_cast<size_t>(Size) + 1]());
  if (!Name || State->NextIsaGetInfoAlt(Isa, HSA_ISA_INFO_NAME, Name.get()) !=
                   HSA_STATUS_SUCCESS)
    return {};
  const char *Terminator = static_cast<const char *>(
      std::memchr(Name.get(), '\0', static_cast<size_t>(Size) + 1));
  if (!Terminator)
    return {};
  const size_t Length = static_cast<size_t>(Terminator - Name.get());
  if (Length != Size && Length + 1 != Size)
    return {};
  return std::string(Name.get(), Length);
}

bool isGpu(hsa_agent_t Agent) {
  hsa_device_type_t Type = HSA_DEVICE_TYPE_CPU;
  return State &&
         State->NextAgentGetInfo(Agent, HSA_AGENT_INFO_DEVICE, &Type) ==
             HSA_STATUS_SUCCESS &&
         Type == HSA_DEVICE_TYPE_GPU;
}

AgentView getAgentView(hsa_agent_t Agent) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->Agents.find(Agent.handle);
  return It == State->Agents.end() ? AgentView{} : It->second;
}

struct MetadataOwner {
  amd_comgr_metadata_node_t Node{};
  ~MetadataOwner() {
    if (Node.handle)
      amd_comgr_destroy_metadata(Node);
  }
};

struct DataOwner {
  amd_comgr_data_t Data{};
  ~DataOwner() {
    if (Data.handle)
      amd_comgr_release_data(Data);
  }
};

struct TranspileResultOwner {
  amd_comgr_hotswap_transpile_result_t Result{};
  ~TranspileResultOwner() {
    if (Result.handle)
      amd_comgr_destroy_hotswap_transpile_result(Result);
  }
};

bool readComgrString(
    llvm::function_ref<amd_comgr_status_t(size_t *, char *)> Query,
    std::string &Value) {
  size_t Size = 0;
  if (Query(&Size, nullptr) != AMD_COMGR_STATUS_SUCCESS || Size == 0)
    return false;
  const size_t Capacity = Size;
  std::unique_ptr<char[]> Buffer(new (std::nothrow) char[Capacity]());
  if (!Buffer || Query(&Size, Buffer.get()) != AMD_COMGR_STATUS_SUCCESS ||
      Size == 0 || Size > Capacity || Buffer[Size - 1] != '\0' ||
      std::memchr(Buffer.get(), '\0', Size - 1) != nullptr)
    return false;
  Value.assign(Buffer.get(), Size - 1);
  return true;
}

bool metadataString(amd_comgr_metadata_node_t Node, std::string &Value) {
  return readComgrString(
      [&](size_t *Size, char *Buffer) {
        return amd_comgr_get_metadata_string(Node, Size, Buffer);
      },
      Value);
}

bool metadataLookupString(amd_comgr_metadata_node_t Map, const char *Key,
                          std::string &Value) {
  MetadataOwner Entry;
  return amd_comgr_metadata_lookup(Map, Key, &Entry.Node) ==
             AMD_COMGR_STATUS_SUCCESS &&
         metadataString(Entry.Node, Value);
}

bool metadataLookupUInt32(amd_comgr_metadata_node_t Map, const char *Key,
                          uint32_t &Value) {
  std::string Text;
  if (!metadataLookupString(Map, Key, Text) || Text.empty())
    return false;
  uint32_t Parsed = 0;
  if (!llvm::to_integer(Text, Parsed, 10))
    return false;
  Value = Parsed;
  return true;
}

bool metadataLookupOptionalBool(amd_comgr_metadata_node_t Map, const char *Key,
                                bool &Value) {
  MetadataOwner Entry;
  if (amd_comgr_metadata_lookup(Map, Key, &Entry.Node) !=
      AMD_COMGR_STATUS_SUCCESS) {
    Value = false;
    return true;
  }
  std::string Text;
  if (!metadataString(Entry.Node, Text))
    return false;
  if (Text == "true" || Text == "1") {
    Value = true;
    return true;
  }
  if (Text == "false" || Text == "0") {
    Value = false;
    return true;
  }
  return false;
}

bool createComgrData(const Bytes &Object, DataOwner &Result) {
  return Object && !Object->empty() &&
         amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Result.Data) ==
             AMD_COMGR_STATUS_SUCCESS &&
         amd_comgr_set_data(Result.Data, Object->size(),
                            reinterpret_cast<const char *>(Object->data())) ==
             AMD_COMGR_STATUS_SUCCESS;
}

struct KernelMetadata {
  std::string Name;
  std::string Symbol;
  uint32_t KernargSegmentSize = 0;
  uint32_t KernargSegmentAlignment = 0;
  uint32_t PrivateSegmentSize = 0;
  uint32_t GroupSegmentSize = 0;
  uint32_t WavefrontSize = 0;
  bool DynamicCallstack = false;
};

struct SourceSymbolInspection {
  const std::vector<KernelMetadata> &Kernels;
  llvm::StringSet<> DefinedKernelDescriptors;
  std::string Failure;
};

amd_comgr_status_t inspectSourceSymbol(amd_comgr_symbol_t Symbol,
                                       void *UserData) {
  auto &Inspection = *static_cast<SourceSymbolInspection *>(UserData);
  amd_comgr_symbol_type_t Type = AMD_COMGR_SYMBOL_TYPE_UNKNOWN;
  bool Undefined = false;
  uint64_t NameLength = 0;
  if (amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_TYPE, &Type) !=
          AMD_COMGR_STATUS_SUCCESS ||
      amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_IS_UNDEFINED,
                                &Undefined) != AMD_COMGR_STATUS_SUCCESS ||
      amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH,
                                &NameLength) != AMD_COMGR_STATUS_SUCCESS ||
      NameLength >= std::numeric_limits<size_t>::max()) {
    Inspection.Failure = "cannot inspect a source object symbol";
    return AMD_COMGR_STATUS_ERROR;
  }
  if (Type != AMD_COMGR_SYMBOL_TYPE_OBJECT &&
      Type != AMD_COMGR_SYMBOL_TYPE_COMMON)
    return AMD_COMGR_STATUS_SUCCESS;

  std::unique_ptr<char[]> Name(
      new (std::nothrow) char[static_cast<size_t>(NameLength) + 1]());
  uint64_t Size = 0;
  if (!Name ||
      amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_NAME,
                                Name.get()) != AMD_COMGR_STATUS_SUCCESS ||
      amd_comgr_symbol_get_info(Symbol, AMD_COMGR_SYMBOL_INFO_SIZE, &Size) !=
          AMD_COMGR_STATUS_SUCCESS ||
      Name[NameLength] != '\0' ||
      std::memchr(Name.get(), '\0', static_cast<size_t>(NameLength)) !=
          nullptr) {
    Inspection.Failure = "cannot inspect source object storage";
    return AMD_COMGR_STATUS_ERROR;
  }

  const llvm::StringRef SymbolName(Name.get(), static_cast<size_t>(NameLength));
  constexpr uint64_t KernelDescriptorSize = 64;
  const bool IsKernelDescriptor =
      !Undefined && Size == KernelDescriptorSize &&
      llvm::any_of(Inspection.Kernels, [&](const KernelMetadata &Kernel) {
        return SymbolName == Kernel.Symbol;
      });
  // Clang emits this one-byte marker solely to associate the host and device
  // images from one HIP compilation unit. Kernels do not use it as shared
  // program state.
  const bool IsHipCompilationUnitMarker =
      !Undefined && Size == 1 && SymbolName.starts_with("__hip_cuid_");
  if (IsKernelDescriptor) {
    Inspection.DefinedKernelDescriptors.insert(SymbolName);
    return AMD_COMGR_STATUS_SUCCESS;
  }
  if (IsHipCompilationUnitMarker)
    return AMD_COMGR_STATUS_SUCCESS;

  Inspection.Failure =
      "source object uses unsupported device storage symbol '" +
      SymbolName.str() + "'";
  return AMD_COMGR_STATUS_ERROR;
}

bool inspectSourceObject(const Bytes &Object, std::string &SourceIsa,
                         std::vector<KernelMetadata> &Kernels,
                         std::string &Failure) {
  DataOwner Input;
  if (!createComgrData(Object, Input)) {
    Failure = "cannot create COMGR source data";
    return false;
  }

  if (!readComgrString(
          [&](size_t *Size, char *Buffer) {
            return amd_comgr_get_data_isa_name(Input.Data, Size, Buffer);
          },
          SourceIsa)) {
    Failure = "COMGR could not read the source ISA";
    return false;
  }

  MetadataOwner Root;
  MetadataOwner KernelList;
  if (amd_comgr_get_data_metadata(Input.Data, &Root.Node) !=
          AMD_COMGR_STATUS_SUCCESS ||
      amd_comgr_metadata_lookup(Root.Node, "amdhsa.kernels",
                                &KernelList.Node) != AMD_COMGR_STATUS_SUCCESS) {
    Failure = "source object has no amdhsa.kernels metadata";
    return false;
  }

  size_t KernelCount = 0;
  if (amd_comgr_get_metadata_list_size(KernelList.Node, &KernelCount) !=
          AMD_COMGR_STATUS_SUCCESS ||
      KernelCount == 0) {
    Failure = "source object has no metadata kernels";
    return false;
  }
  Kernels.reserve(KernelCount);
  for (size_t I = 0; I != KernelCount; ++I) {
    MetadataOwner Kernel;
    KernelMetadata Metadata;
    const auto RejectMetadata = [&](llvm::StringRef Detail) {
      Failure = "source kernel metadata entry " + std::to_string(I) + " " +
                Detail.str();
      return false;
    };
    if (amd_comgr_index_list_metadata(KernelList.Node, I, &Kernel.Node) !=
        AMD_COMGR_STATUS_SUCCESS)
      return RejectMetadata("cannot be read");
    if (!metadataLookupString(Kernel.Node, ".name", Metadata.Name))
      return RejectMetadata("has an invalid .name");
    if (!metadataLookupString(Kernel.Node, ".symbol", Metadata.Symbol))
      return RejectMetadata("has an invalid .symbol");
    if (!metadataLookupUInt32(Kernel.Node, ".kernarg_segment_size",
                              Metadata.KernargSegmentSize))
      return RejectMetadata("has an invalid .kernarg_segment_size");
    if (!metadataLookupUInt32(Kernel.Node, ".kernarg_segment_align",
                              Metadata.KernargSegmentAlignment))
      return RejectMetadata("has an invalid .kernarg_segment_align");
    if (!metadataLookupUInt32(Kernel.Node, ".private_segment_fixed_size",
                              Metadata.PrivateSegmentSize))
      return RejectMetadata("has an invalid .private_segment_fixed_size");
    if (!metadataLookupUInt32(Kernel.Node, ".group_segment_fixed_size",
                              Metadata.GroupSegmentSize))
      return RejectMetadata("has an invalid .group_segment_fixed_size");
    if (!metadataLookupUInt32(Kernel.Node, ".wavefront_size",
                              Metadata.WavefrontSize))
      return RejectMetadata("has an invalid .wavefront_size");
    if (!metadataLookupOptionalBool(Kernel.Node, ".uses_dynamic_stack",
                                    Metadata.DynamicCallstack))
      return RejectMetadata("has an invalid .uses_dynamic_stack");
    if (Metadata.Name.empty())
      return RejectMetadata("has an empty .name");
    if (Metadata.Symbol.empty())
      return RejectMetadata("has an empty .symbol");
    if (Metadata.Symbol.size() > std::numeric_limits<uint32_t>::max())
      return RejectMetadata("has a .symbol longer than the HSA ABI permits");
    if (Metadata.KernargSegmentAlignment == 0 ||
        (Metadata.KernargSegmentAlignment &
         (Metadata.KernargSegmentAlignment - 1)) != 0)
      return RejectMetadata("has a non-power-of-two .kernarg_segment_align");
    if (Metadata.WavefrontSize != 32 && Metadata.WavefrontSize != 64)
      return RejectMetadata("has an unsupported .wavefront_size");
    Kernels.push_back(std::move(Metadata));
  }

  SourceSymbolInspection Inspection{Kernels, {}, {}};
  const amd_comgr_status_t SymbolStatus =
      amd_comgr_iterate_symbols(Input.Data, inspectSourceSymbol, &Inspection);
  if (SymbolStatus != AMD_COMGR_STATUS_SUCCESS || !Inspection.Failure.empty()) {
    Failure = Inspection.Failure.empty()
                  ? "COMGR could not inspect source object symbols"
                  : std::move(Inspection.Failure);
    return false;
  }
  for (const KernelMetadata &Kernel : Kernels) {
    if (!Inspection.DefinedKernelDescriptors.contains(Kernel.Symbol)) {
      Failure = "source object has no defined descriptor for kernel '" +
                Kernel.Name + "'";
      return false;
    }
  }
  std::vector<std::string> KernelDescriptors;
  KernelDescriptors.reserve(Kernels.size());
  for (const KernelMetadata &Kernel : Kernels)
    KernelDescriptors.push_back(Kernel.Symbol);
  if (!inspectSourceStorage(*Object, KernelDescriptors, Failure))
    return false;
  return true;
}

std::string resultString(amd_comgr_hotswap_transpile_result_t Result,
                         amd_comgr_hotswap_transpile_result_string_t Field) {
  if (!Result.handle)
    return {};
  std::string Value;
  if (!readComgrString(
          [&](size_t *Size, char *Buffer) {
            return amd_comgr_hotswap_transpile_result_get_string(Result, Field,
                                                                 Size, Buffer);
          },
          Value))
    return {};
  return Value;
}

} // namespace
} // namespace COMGR::hotswap::hsa_tool

namespace COMGR::hotswap::hsa_tool {
namespace {

hsa_status_t HSA_API toolAgentGetInfo(hsa_agent_t Agent,
                                      hsa_agent_info_t Attribute, void *Value) {
  if (!Value)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const AgentView View = getAgentView(Agent);
  if (!View.Agent.handle)
    return State->NextAgentGetInfo(Agent, Attribute, Value);
  switch (Attribute) {
  case HSA_AGENT_INFO_ISA:
    *static_cast<hsa_isa_t *>(Value) = View.PresentedIsa;
    return HSA_STATUS_SUCCESS;
  case HSA_AGENT_INFO_NAME: {
    // HSA_AGENT_INFO_NAME is specified as a zero-filled char[64].
    std::memset(Value, 0, AgentNameSize);
    std::memcpy(Value, View.PresentedGfx.data(), View.PresentedGfx.size());
    return HSA_STATUS_SUCCESS;
  }
  case HSA_AGENT_INFO_WAVEFRONT_SIZE:
    *static_cast<uint32_t *>(Value) = View.WavefrontSize;
    return HSA_STATUS_SUCCESS;
  default:
    // In particular, HSA_AMD_AGENT_INFO_EXECUTION_ISA must remain the physical
    // query implemented by ROCr. It is never synthesized from presentation.
    return State->NextAgentGetInfo(Agent, Attribute, Value);
  }
}

hsa_status_t HSA_API toolAgentIterateIsas(hsa_agent_t Agent,
                                          hsa_status_t (*Callback)(hsa_isa_t,
                                                                   void *),
                                          void *Data) {
  if (!Callback)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const AgentView View = getAgentView(Agent);
  if (!View.Agent.handle)
    return State->NextAgentIterateIsas(Agent, Callback, Data);
  return Callback(View.PresentedIsa, Data);
}

hsa_status_t firstWavefront(hsa_wavefront_t Wave, void *Data) {
  *static_cast<hsa_wavefront_t *>(Data) = Wave;
  return HSA_STATUS_INFO_BREAK;
}

hsa_status_t discoverAgent(hsa_agent_t Agent, void *) {
  if (!isGpu(Agent))
    return HSA_STATUS_SUCCESS;

  AgentView View;
  View.Agent = Agent;
  View.PresentedIsa = State->PresentedIsa;
  View.PresentedName = State->PresentedName;
  View.PresentedGfx = State->PresentedGfx;
  if (State->NextAgentGetInfo(
          Agent,
          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_EXECUTION_ISA),
          &View.ExecutionIsa) != HSA_STATUS_SUCCESS ||
      !View.ExecutionIsa.handle) {
    llvm::errs() << "hotswap-hsa-tool: ROCr does not provide the required "
                    "physical execution-ISA query\n";
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  }
  View.ExecutionName = isaName(View.ExecutionIsa);
  View.ExecutionGfx = processor(View.ExecutionName);
  if (View.ExecutionName.empty() || View.ExecutionGfx.empty())
    return HSA_STATUS_ERROR_INVALID_ISA;

  hsa_wavefront_t Wave{};
  const hsa_status_t WaveStatus =
      State->NextIsaIterateWavefronts(View.PresentedIsa, firstWavefront, &Wave);
  if ((WaveStatus != HSA_STATUS_SUCCESS &&
       WaveStatus != HSA_STATUS_INFO_BREAK) ||
      !Wave.handle ||
      State->NextWavefrontGetInfo(Wave, HSA_WAVEFRONT_INFO_SIZE,
                                  &View.WavefrontSize) != HSA_STATUS_SUCCESS ||
      (View.WavefrontSize != 32 && View.WavefrontSize != 64))
    return HSA_STATUS_ERROR_INVALID_ISA;
  if (State->NextIsaGetInfoAlt(
          View.ExecutionIsa, HSA_ISA_INFO_WORKGROUP_MAX_DIM,
          View.MaxWorkgroupDim.data()) != HSA_STATUS_SUCCESS ||
      State->NextIsaGetInfoAlt(View.ExecutionIsa,
                               HSA_ISA_INFO_WORKGROUP_MAX_SIZE,
                               &View.MaxWorkgroupSize) != HSA_STATUS_SUCCESS ||
      State->NextIsaGetInfoAlt(View.ExecutionIsa, HSA_ISA_INFO_GRID_MAX_DIM,
                               &View.MaxGridDim) != HSA_STATUS_SUCCESS ||
      State->NextIsaGetInfoAlt(View.ExecutionIsa, HSA_ISA_INFO_GRID_MAX_SIZE,
                               &View.MaxGridSize) != HSA_STATUS_SUCCESS ||
      View.MaxWorkgroupDim[0] == 0 || View.MaxWorkgroupDim[1] == 0 ||
      View.MaxWorkgroupDim[2] == 0 || View.MaxWorkgroupSize == 0 ||
      View.MaxGridDim.x == 0 || View.MaxGridDim.y == 0 ||
      View.MaxGridDim.z == 0 || View.MaxGridSize == 0)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  View.NeedsTranslation = View.PresentedName != View.ExecutionName;
  std::lock_guard<std::mutex> Lock(State->Mutex);
  State->Agents[Agent.handle] = std::move(View);
  return HSA_STATUS_SUCCESS;
}

std::shared_ptr<KernelRecord> kernelForToken(const void *Address) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->KernelTokens.find(reinterpret_cast<uint64_t>(Address));
  if (It == State->KernelTokens.end())
    return {};
  const std::shared_ptr<SourceObject> Object = It->second->Object.lock();
  if (!Object || !Object->Alive.load(std::memory_order_acquire))
    return {};
  return It->second;
}

hsa_status_t toolLoaderHostAddress(const void *DeviceAddress,
                                   const void **HostAddress) {
  if (!HostAddress)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *HostAddress = nullptr;
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  std::shared_ptr<KernelRecord> Kernel = kernelForToken(DeviceAddress);
  if (!Kernel)
    return State->NextLoaderHostAddress(DeviceAddress, HostAddress);

  // A virtual token is deliberately not a loader allocation, and there is no
  // contract-preserving host descriptor to return before translation.  In
  // particular, a source descriptor would describe the wrong execution ISA.
  // Do not turn loader introspection into eager translation of every kernel.
  std::lock_guard<std::mutex> Lock(Kernel->Mutex);
  if (!Kernel->Attempted)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!Kernel->Succeeded)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  return State->NextLoaderHostAddress(
      reinterpret_cast<const void *>(Kernel->Target.KernelObject), HostAddress);
}

hsa_status_t toolLoaderExecutable(const void *DeviceAddress,
                                  hsa_executable_t *Executable) {
  if (!Executable)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  std::shared_ptr<KernelRecord> Kernel = kernelForToken(DeviceAddress);
  if (!Kernel)
    return State->NextLoaderExecutable(DeviceAddress, Executable);
  const std::shared_ptr<SourceObject> Object = Kernel->Object.lock();
  if (!Object)
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  *Executable = Object->Parent;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t
toolLoaderQuerySegments(hsa_ven_amd_loader_segment_descriptor_t *Descriptors,
                        size_t *Count) {
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (!State->Executables.empty())
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  }
  return State->NextLoaderQuerySegments(Descriptors, Count);
}

hsa_status_t toolLoaderIterateLoaded(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_loaded_code_object_t,
                             void *),
    void *Data) {
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (State->Executables.find(Executable.handle) != State->Executables.end())
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  }
  return State->NextLoaderIterateLoaded(Executable, Callback, Data);
}

hsa_status_t
toolLoaderIterateExecutables(hsa_status_t (*Callback)(hsa_executable_t, void *),
                             void *Data) {
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (!State->Executables.empty())
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  }
  return State->NextLoaderIterateExecutables(Callback, Data);
}

template <typename FunctionT>
void patchLoaderFunction(FunctionT &Slot, FunctionT Replacement,
                         FunctionT &Next, const char *Description) {
  if (!Slot)
    refuse(std::string("loader ") + Description + " table entry is null");
  if (!Next)
    Next = Slot;
  else if (Next != Slot && Slot != Replacement)
    refuse(std::string("loader ") + Description +
           " table changed while the tool was active");
  Slot = Replacement;
}

template <typename TableT> void patchLoaderBase(TableT &Table) {
  patchLoaderFunction(Table.hsa_ven_amd_loader_query_host_address,
                      toolLoaderHostAddress, State->NextLoaderHostAddress,
                      "host-address");
  patchLoaderFunction(Table.hsa_ven_amd_loader_query_segment_descriptors,
                      toolLoaderQuerySegments, State->NextLoaderQuerySegments,
                      "segment");
  patchLoaderFunction(Table.hsa_ven_amd_loader_query_executable,
                      toolLoaderExecutable, State->NextLoaderExecutable,
                      "executable");
}

void patchLoaderTable(hsa_ven_amd_loader_1_00_pfn_t &Table) {
  patchLoaderBase(Table);
}

void patchLoaderTable(hsa_ven_amd_loader_1_01_pfn_t &Table) {
  patchLoaderBase(Table);
  patchLoaderFunction(
      Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects,
      toolLoaderIterateLoaded, State->NextLoaderIterateLoaded,
      "code-object iteration");
}

void patchLoaderTable(hsa_ven_amd_loader_1_02_pfn_t &Table) {
  patchLoaderBase(Table);
  patchLoaderFunction(
      Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects,
      toolLoaderIterateLoaded, State->NextLoaderIterateLoaded,
      "code-object iteration");
  patchLoaderFunction(
      Table
          .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
      toolReaderFileSlice, State->NextLoaderReaderFileSlice,
      "file-slice reader");
}

void patchLoaderTable(hsa_ven_amd_loader_1_03_pfn_t &Table) {
  patchLoaderBase(Table);
  patchLoaderFunction(
      Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects,
      toolLoaderIterateLoaded, State->NextLoaderIterateLoaded,
      "code-object iteration");
  patchLoaderFunction(
      Table
          .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
      toolReaderFileSlice, State->NextLoaderReaderFileSlice,
      "file-slice reader");
  patchLoaderFunction(Table.hsa_ven_amd_loader_iterate_executables,
                      toolLoaderIterateExecutables,
                      State->NextLoaderIterateExecutables,
                      "executable iteration");
}

template <typename FunctionT>
void patchRawLoaderFunction(void *RawTable, size_t TableSize, size_t Offset,
                            FunctionT Replacement, FunctionT &Next,
                            const char *Description) {
  static_assert(std::is_trivially_copyable_v<FunctionT>);
  if (Offset > TableSize || sizeof(FunctionT) > TableSize - Offset)
    return;
  llvm::MutableArrayRef<unsigned char> TableBytes(
      static_cast<unsigned char *>(RawTable), TableSize);
  llvm::MutableArrayRef<unsigned char> SlotBytes =
      TableBytes.drop_front(Offset).take_front(sizeof(FunctionT));
  FunctionT Slot;
  std::memcpy(static_cast<void *>(&Slot), SlotBytes.data(), sizeof(Slot));
  patchLoaderFunction(Slot, Replacement, Next, Description);
  std::memcpy(SlotBytes.data(), static_cast<const void *>(&Slot), sizeof(Slot));
}

static_assert(offsetof(hsa_ven_amd_loader_1_00_pfn_t,
                       hsa_ven_amd_loader_query_executable) ==
              offsetof(hsa_ven_amd_loader_1_03_pfn_t,
                       hsa_ven_amd_loader_query_executable));
static_assert(
    offsetof(hsa_ven_amd_loader_1_01_pfn_t,
             hsa_ven_amd_loader_executable_iterate_loaded_code_objects) ==
    offsetof(hsa_ven_amd_loader_1_03_pfn_t,
             hsa_ven_amd_loader_executable_iterate_loaded_code_objects));
static_assert(
    offsetof(
        hsa_ven_amd_loader_1_02_pfn_t,
        hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size) ==
    offsetof(
        hsa_ven_amd_loader_1_03_pfn_t,
        hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size));

#define PATCH_RAW_LOADER_FUNCTION(Member, Replacement, Next, Description)      \
  patchRawLoaderFunction(RawTable, TableSize,                                  \
                         offsetof(hsa_ven_amd_loader_1_03_pfn_t, Member),      \
                         Replacement, Next, Description)

void patchRawLoaderTable(size_t TableSize, void *RawTable) {
  if (!RawTable)
    return;
  PATCH_RAW_LOADER_FUNCTION(hsa_ven_amd_loader_query_host_address,
                            toolLoaderHostAddress, State->NextLoaderHostAddress,
                            "host-address");
  PATCH_RAW_LOADER_FUNCTION(hsa_ven_amd_loader_query_segment_descriptors,
                            toolLoaderQuerySegments,
                            State->NextLoaderQuerySegments, "segment");
  PATCH_RAW_LOADER_FUNCTION(hsa_ven_amd_loader_query_executable,
                            toolLoaderExecutable, State->NextLoaderExecutable,
                            "executable");
  PATCH_RAW_LOADER_FUNCTION(
      hsa_ven_amd_loader_executable_iterate_loaded_code_objects,
      toolLoaderIterateLoaded, State->NextLoaderIterateLoaded,
      "code-object iteration");
  PATCH_RAW_LOADER_FUNCTION(
      hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
      toolReaderFileSlice, State->NextLoaderReaderFileSlice,
      "file-slice reader");
  PATCH_RAW_LOADER_FUNCTION(
      hsa_ven_amd_loader_iterate_executables, toolLoaderIterateExecutables,
      State->NextLoaderIterateExecutables, "executable iteration");
}

#undef PATCH_RAW_LOADER_FUNCTION

hsa_status_t HSA_API toolGetExtensionTable(uint16_t Extension, uint16_t Major,
                                           uint16_t Minor, void *Table) {
  const hsa_status_t Status =
      State->NextGetExtensionTable(Extension, Major, Minor, Table);
  if (Status == HSA_STATUS_SUCCESS && Extension == HSA_EXTENSION_AMD_LOADER &&
      Major == 1) {
    if (!Table)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> Lock(State->LoaderTableMutex);
    switch (Minor) {
    case 0:
      patchLoaderTable(*static_cast<hsa_ven_amd_loader_1_00_pfn_t *>(Table));
      break;
    case 1:
      patchLoaderTable(*static_cast<hsa_ven_amd_loader_1_01_pfn_t *>(Table));
      break;
    case 2:
      patchLoaderTable(*static_cast<hsa_ven_amd_loader_1_02_pfn_t *>(Table));
      break;
    case 3:
      patchLoaderTable(*static_cast<hsa_ven_amd_loader_1_03_pfn_t *>(Table));
      break;
    default:
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
    }
  }
  return Status;
}

hsa_status_t HSA_API toolGetMajorExtensionTable(uint16_t Extension,
                                                uint16_t Major,
                                                size_t TableSize, void *Table) {
  if (Extension == HSA_EXTENSION_AMD_LOADER && Major == 1 &&
      TableSize > sizeof(hsa_ven_amd_loader_1_03_pfn_t))
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  const hsa_status_t Status =
      State->NextGetMajorExtensionTable(Extension, Major, TableSize, Table);
  if (Status == HSA_STATUS_SUCCESS && Extension == HSA_EXTENSION_AMD_LOADER &&
      Major == 1) {
    if (!Table)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> Lock(State->LoaderTableMutex);
    patchRawLoaderTable(TableSize, Table);
  }
  return Status;
}

void patchKernel(KernelDispatchPacket &Packet) {
  std::shared_ptr<KernelRecord> Kernel =
      kernelForToken(reinterpret_cast<const void *>(Packet.KernelObject));
  if (!Kernel) {
    const std::string Reason =
        "unregistered kernel_object " + std::to_string(Packet.KernelObject);
    proofOrRefuse("\"event\":\"dispatch_rejected\",\"reason\":\"" +
                  jsonEscape(Reason) + "\"");
    refuse(Reason);
  }

  if (!ensureTranslated(Kernel)) {
    const std::string Reason = "translation of " + Kernel->MetadataName +
                               " failed: " + Kernel->Failure;
    proofOrRefuse("\"event\":\"dispatch_rejected\",\"kernel\":\"" +
                  jsonEscape(Kernel->MetadataName) + "\",\"reason\":\"" +
                  jsonEscape(Reason) + "\"");
    refuse(Reason);
  }

  const uint64_t SourceObject = Packet.KernelObject;
  const DispatchRewriteError Error =
      rewriteKernelDispatch(Kernel->Target, Packet);
  if (Error != DispatchRewriteError::None) {
    const std::string Reason = std::string(dispatchRewriteErrorString(Error)) +
                               " for " + Kernel->MetadataName;
    proofOrRefuse("\"event\":\"dispatch_rejected\",\"kernel\":\"" +
                  jsonEscape(Kernel->MetadataName) + "\",\"reason\":\"" +
                  jsonEscape(Reason) + "\"");
    refuse(Reason);
  }

  State->Count.RewrittenDispatches.fetch_add(1, std::memory_order_relaxed);
  proofOrRefuse(
      "\"event\":\"dispatch_rewritten\",\"kernel\":\"" +
      jsonEscape(Kernel->MetadataName) +
      "\",\"source_kernel_token\":" + std::to_string(SourceObject) +
      ",\"target_kernel_object\":" + std::to_string(Packet.KernelObject));
}

void recordDispatchIntercepted(uint64_t KernelObject) {
  State->Count.InterceptedDispatches.fetch_add(1, std::memory_order_relaxed);
  proofOrRefuse("\"event\":\"dispatch_intercepted\","
                "\"source_kernel_object\":" +
                std::to_string(KernelObject));
}

[[noreturn]] void refusePacket(const std::string &Reason);

void validatePacketHeader(uint16_t Header) {
  constexpr uint16_t TypeMask = ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1)
                                << HSA_PACKET_HEADER_TYPE;
  constexpr uint16_t BarrierMask = ((1u << HSA_PACKET_HEADER_WIDTH_BARRIER) - 1)
                                   << HSA_PACKET_HEADER_BARRIER;
  constexpr uint16_t AcquireMask =
      ((1u << HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE) - 1)
      << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  constexpr uint16_t ReleaseMask =
      ((1u << HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE) - 1)
      << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  constexpr uint16_t KnownMask =
      TypeMask | BarrierMask | AcquireMask | ReleaseMask;
  if (Header & ~KnownMask)
    refusePacket("AQL packet header has nonzero reserved bits");

  const uint16_t Acquire =
      (Header & AcquireMask) >> HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
  const uint16_t Release =
      (Header & ReleaseMask) >> HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
  if (Acquire > HSA_FENCE_SCOPE_SYSTEM || Release > HSA_FENCE_SCOPE_SYSTEM)
    refusePacket("AQL packet header has an invalid fence scope");
}

void validateDispatchShape(uint16_t Setup,
                           const KernelDispatchPacket &Dispatch) {
  constexpr uint16_t DimensionMask =
      ((1u << HSA_KERNEL_DISPATCH_PACKET_SETUP_WIDTH_DIMENSIONS) - 1)
      << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  if (Setup & ~DimensionMask)
    refusePacket("kernel dispatch setup has nonzero reserved bits");
  const uint16_t Dimensions =
      (Setup & DimensionMask) >> HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  if (Dimensions < 1 || Dimensions > 3)
    refusePacket("kernel dispatch has an invalid dimension count");
  if (!Dispatch.Grid)
    refusePacket("kernel dispatch does not define a grid");
  if (!Dispatch.WorkgroupSizeX || !Dispatch.WorkgroupSizeY ||
      !Dispatch.WorkgroupSizeZ || !Dispatch.Grid->X || !Dispatch.Grid->Y ||
      !Dispatch.Grid->Z)
    refusePacket("kernel dispatch has an invalid zero dimension");
  if (Dispatch.Grid->X < Dispatch.WorkgroupSizeX ||
      Dispatch.Grid->Y < Dispatch.WorkgroupSizeY ||
      Dispatch.Grid->Z < Dispatch.WorkgroupSizeZ)
    refusePacket("kernel dispatch grid is smaller than its workgroup");
  if ((Dimensions == 1 &&
       (Dispatch.WorkgroupSizeY != 1 || Dispatch.Grid->Y != 1)) ||
      (Dimensions <= 2 &&
       (Dispatch.WorkgroupSizeZ != 1 || Dispatch.Grid->Z != 1)))
    refusePacket("kernel dispatch has non-unit inactive dimensions");
}

void patchDispatch(hsa_kernel_dispatch_packet_t &Packet) {
  recordDispatchIntercepted(Packet.kernel_object);
  if (Packet.reserved0 != 0)
    refusePacket("kernel dispatch has a nonzero reserved field");
  KernelDispatchPacket Dispatch{
      Packet.kernel_object,
      Packet.private_segment_size,
      Packet.group_segment_size,
      Packet.workgroup_size_x,
      Packet.workgroup_size_y,
      Packet.workgroup_size_z,
      DispatchGrid{Packet.grid_size_x, Packet.grid_size_y, Packet.grid_size_z}};
  validateDispatchShape(Packet.setup, Dispatch);
  patchKernel(Dispatch);
  Packet.kernel_object = Dispatch.KernelObject;
  Packet.private_segment_size = Dispatch.PrivateSegmentSize;
  Packet.group_segment_size = Dispatch.GroupSegmentSize;
  Packet.workgroup_size_x = Dispatch.WorkgroupSizeX;
  Packet.grid_size_x = Dispatch.Grid->X;
}

[[noreturn]] void refusePacket(const std::string &Reason) {
  proofOrRefuse("\"event\":\"packet_rejected\",\"reason\":\"" +
                jsonEscape(Reason) + "\"");
  refuse(Reason);
}

void lowerExtendedDispatch(hsa_kernel_dispatch_packet_t &Storage) {
  hsa_amd_ext_kernel_dispatch_packet_t Packet{};
  std::memcpy(&Packet, &Storage, sizeof(Packet));
  recordDispatchIntercepted(Packet.kernel_object);
  if (Packet.reserved0 != 0 || Packet.dep_signal.handle != 0 ||
      Packet.perf_hint.hint_val != 0)
    refusePacket("extended dispatch uses unsupported reserved, dependency, or "
                 "performance-hint fields");
  if (Packet.cluster_size_x != 1 || Packet.cluster_size_y != 1 ||
      Packet.cluster_size_z != 1)
    refusePacket("clustered extended dispatch is unsupported on the target");
  if (!Packet.workgroup_size_x || !Packet.workgroup_size_y ||
      !Packet.workgroup_size_z || !Packet.cluster_count_x ||
      !Packet.cluster_count_y || !Packet.cluster_count_z)
    refusePacket("extended dispatch has an invalid zero dimension");

  const uint64_t UnscaledGridX =
      static_cast<uint64_t>(Packet.cluster_count_x) * Packet.workgroup_size_x;
  const uint64_t UnscaledGridY =
      static_cast<uint64_t>(Packet.cluster_count_y) * Packet.workgroup_size_y;
  const uint64_t UnscaledGridZ =
      static_cast<uint64_t>(Packet.cluster_count_z) * Packet.workgroup_size_z;
  if (UnscaledGridX > std::numeric_limits<uint32_t>::max() ||
      UnscaledGridY > std::numeric_limits<uint32_t>::max() ||
      UnscaledGridZ > std::numeric_limits<uint32_t>::max())
    refusePacket("lowered extended dispatch grid overflows");
  uint32_t GridX = static_cast<uint32_t>(UnscaledGridX);
  const uint32_t GridY = static_cast<uint32_t>(UnscaledGridY);
  const uint32_t GridZ = static_cast<uint32_t>(UnscaledGridZ);

  KernelDispatchPacket Dispatch{Packet.kernel_object,
                                Packet.private_segment_size,
                                Packet.group_segment_size,
                                Packet.workgroup_size_x,
                                Packet.workgroup_size_y,
                                Packet.workgroup_size_z,
                                DispatchGrid{GridX, GridY, GridZ}};
  validateDispatchShape(Packet.setup, Dispatch);
  patchKernel(Dispatch);
  Packet.kernel_object = Dispatch.KernelObject;
  Packet.private_segment_size = Dispatch.PrivateSegmentSize;
  Packet.group_segment_size = Dispatch.GroupSegmentSize;
  Packet.workgroup_size_x = Dispatch.WorkgroupSizeX;
  GridX = Dispatch.Grid->X;

  hsa_kernel_dispatch_packet_t Lowered{};
  constexpr uint16_t TypeMask = ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1)
                                << HSA_PACKET_HEADER_TYPE;
  Lowered.header = (Packet.header & ~TypeMask) |
                   (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
  Lowered.setup = Packet.setup;
  Lowered.workgroup_size_x = Packet.workgroup_size_x;
  Lowered.workgroup_size_y = Packet.workgroup_size_y;
  Lowered.workgroup_size_z = Packet.workgroup_size_z;
  Lowered.grid_size_x = GridX;
  Lowered.grid_size_y = GridY;
  Lowered.grid_size_z = GridZ;
  Lowered.private_segment_size = Packet.private_segment_size;
  Lowered.group_segment_size = Packet.group_segment_size;
  Lowered.kernel_object = Packet.kernel_object;
  Lowered.kernarg_address = Packet.kernarg_address;
  Lowered.completion_signal = Packet.completion_signal;
  Storage = Lowered;
  proofOrRefuse("\"event\":\"extended_dispatch_lowered\"");
}

void interceptPackets(const void *Packets, uint64_t Count, uint64_t,
                      void *CallbackData,
                      hsa_amd_queue_intercept_packet_writer Writer) {
  const auto *Queue = static_cast<const ProtectedQueueRecord *>(CallbackData);
  if (!Packets || !Writer || !Queue || Count > Queue->Capacity ||
      Count > static_cast<uint64_t>(std::numeric_limits<ptrdiff_t>::max()) /
                  sizeof(hsa_kernel_dispatch_packet_t))
    refusePacket("invalid queue-intercept callback arguments");

  // Keep every translated child used by this batch alive until ROCr has
  // accepted the rewritten packets. Executable destruction takes the
  // exclusive side of this lock.
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  std::unique_ptr<hsa_kernel_dispatch_packet_t[]> Copy;
  if (Count) {
    Copy.reset(new (std::nothrow)
                   hsa_kernel_dispatch_packet_t[static_cast<size_t>(Count)]);
    if (!Copy)
      refusePacket("cannot allocate intercepted packet batch");
    std::memcpy(Copy.get(), Packets,
                static_cast<size_t>(Count) *
                    sizeof(hsa_kernel_dispatch_packet_t));
  }
  for (size_t I = 0; I != static_cast<size_t>(Count); ++I) {
    hsa_kernel_dispatch_packet_t &Packet = Copy[I];
    validatePacketHeader(Packet.header);
    const uint16_t Type = (Packet.header >> HSA_PACKET_HEADER_TYPE) &
                          ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1);
    if (Type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      patchDispatch(Packet);
      continue;
    }
    if (Type == HSA_PACKET_TYPE_BARRIER_AND ||
        Type == HSA_PACKET_TYPE_BARRIER_OR)
      continue;
    if (Type == HSA_PACKET_TYPE_VENDOR_SPECIFIC) {
      hsa_amd_vendor_packet_header_t Vendor{};
      static_assert(sizeof(Vendor) <= sizeof(Packet));
      std::memcpy(&Vendor, &Packet, sizeof(Vendor));
      if (Vendor.AmdFormat == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
        lowerExtendedDispatch(Packet);
      else if (Vendor.AmdFormat == HSA_AMD_PACKET_TYPE_BARRIER_VALUE)
        continue;
      else
        refusePacket("unsupported vendor packet format " +
                     std::to_string(Vendor.AmdFormat));
      continue;
    }
    refusePacket("unsupported AQL packet type " + std::to_string(Type));
  }
  Writer(Copy.get(), Count);
}

hsa_status_t protectQueue(hsa_queue_t *Queue) {
  if (!Queue || Queue->size == 0)
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  std::unique_ptr<ProtectedQueueRecord> Record(
      new (std::nothrow) ProtectedQueueRecord{Queue, Queue->size});
  if (!Record)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  const hsa_status_t Status =
      State->NextInterceptRegister(Queue, interceptPackets, Record.get());
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (!State->ProtectedQueues.try_emplace(Queue, std::move(Record)).second)
      refuse("runtime reused a live protected queue address");
  }
  State->Count.ProtectedQueues.fetch_add(1, std::memory_order_relaxed);
  proofOrRefuse("\"event\":\"queue_protected\",\"queue\":" +
                std::to_string(reinterpret_cast<uintptr_t>(Queue)));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t
createProtectedQueue(hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
                     void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                     void *Data, uint32_t PrivateSegmentSize,
                     uint32_t GroupSegmentSize, hsa_queue_t **Queue) {
  if (!Queue)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (Type != HSA_QUEUE_TYPE_MULTI)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  hsa_status_t Status =
      State->NextInterceptCreate(Agent, Size, Type, Callback, Data,
                                 PrivateSegmentSize, GroupSegmentSize, Queue);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  Status = protectQueue(*Queue);
  if (Status != HSA_STATUS_SUCCESS) {
    if (State->NextQueueDestroy(*Queue) != HSA_STATUS_SUCCESS)
      refuse("cannot destroy a queue after interception setup failed");
    *Queue = nullptr;
  }
  return Status;
}

hsa_status_t HSA_API
toolQueueCreate(hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
                void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                void *Data, uint32_t PrivateSegmentSize,
                uint32_t GroupSegmentSize, hsa_queue_t **Queue) {
  if (!getAgentView(Agent).NeedsTranslation)
    return State->NextQueueCreate(Agent, Size, Type, Callback, Data,
                                  PrivateSegmentSize, GroupSegmentSize, Queue);
  return createProtectedQueue(Agent, Size, Type, Callback, Data,
                              PrivateSegmentSize, GroupSegmentSize, Queue);
}

hsa_status_t HSA_API toolSoftQueueCreate(hsa_region_t Region, uint32_t Size,
                                         hsa_queue_type32_t Type,
                                         uint32_t Features,
                                         hsa_signal_t DoorbellSignal,
                                         hsa_queue_t **Queue) {
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (llvm::any_of(State->Agents, [](const auto &Entry) {
          return Entry.second.NeedsTranslation;
        }))
      return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  }
  return State->NextSoftQueueCreate(Region, Size, Type, Features,
                                    DoorbellSignal, Queue);
}

hsa_status_t
toolInterceptCreate(hsa_agent_t Agent, uint32_t Size, hsa_queue_type32_t Type,
                    void (*Callback)(hsa_status_t, hsa_queue_t *, void *),
                    void *Data, uint32_t PrivateSegmentSize,
                    uint32_t GroupSegmentSize, hsa_queue_t **Queue) {
  if (!getAgentView(Agent).NeedsTranslation)
    return State->NextInterceptCreate(Agent, Size, Type, Callback, Data,
                                      PrivateSegmentSize, GroupSegmentSize,
                                      Queue);
  return createProtectedQueue(Agent, Size, Type, Callback, Data,
                              PrivateSegmentSize, GroupSegmentSize, Queue);
}

bool allZero(const void *Data, size_t Size) {
  const llvm::ArrayRef<uint8_t> Bytes(static_cast<const uint8_t *>(Data), Size);
  return llvm::all_of(Bytes, [](uint8_t Byte) { return Byte == 0; });
}

hsa_status_t toolAmdQueueCreate(hsa_agent_t Agent,
                                hsa_amd_queue_create_desc_t *Descriptors,
                                uint32_t Count) {
  if (!getAgentView(Agent).NeedsTranslation)
    return State->NextAmdQueueCreate(Agent, Descriptors, Count);
  if (!Descriptors || Count == 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if constexpr (std::numeric_limits<uint32_t>::max() >
                std::numeric_limits<size_t>::max() / sizeof(*Descriptors))
    if (Count > std::numeric_limits<size_t>::max() / sizeof(*Descriptors))
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_status_t FirstError = HSA_STATUS_SUCCESS;
  llvm::MutableArrayRef<hsa_amd_queue_create_desc_t> QueueDescriptors(
      Descriptors, Count);
  for (hsa_amd_queue_create_desc_t &Descriptor : QueueDescriptors) {
    Descriptor.queue = nullptr;
    if (Descriptor.version != HSA_AMD_QUEUE_CREATE_DESC_VERSION ||
        Descriptor.flags != 0 ||
        Descriptor.engine_type != HSA_AMD_QUEUE_ENGINE_COMPUTE ||
        !allZero(Descriptor.reserved_header,
                 sizeof(Descriptor.reserved_header)) ||
        Descriptor.traffic_class != 0 ||
        !allZero(Descriptor.reserved, sizeof(Descriptor.reserved)) ||
        !allZero(Descriptor.engine.compute.reserved,
                 sizeof(Descriptor.engine.compute.reserved)) ||
        Descriptor.queue_size_bytes == 0 ||
        Descriptor.queue_size_bytes % sizeof(hsa_kernel_dispatch_packet_t) !=
            0 ||
        Descriptor.engine.compute.type != HSA_QUEUE_TYPE_MULTI) {
      if (FirstError == HSA_STATUS_SUCCESS)
        FirstError = HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
      continue;
    }

    const uint32_t QueueSize =
        Descriptor.queue_size_bytes / sizeof(hsa_kernel_dispatch_packet_t);
    hsa_status_t Status = createProtectedQueue(
        Agent, QueueSize, Descriptor.engine.compute.type, Descriptor.callback,
        Descriptor.callback_data,
        Descriptor.engine.compute.private_segment_size,
        std::numeric_limits<uint32_t>::max(), &Descriptor.queue);
    if (Status == HSA_STATUS_SUCCESS &&
        Descriptor.priority != HSA_AMD_QUEUE_PRIORITY_NORMAL)
      Status =
          State->NextQueueSetPriority(Descriptor.queue, Descriptor.priority);
    if (Status == HSA_STATUS_SUCCESS &&
        Descriptor.engine.compute.cu_mask_count != 0)
      Status = State->NextQueueSetCuMask(
          Descriptor.queue, Descriptor.engine.compute.cu_mask_count,
          Descriptor.engine.compute.cu_mask);
    if (Status != HSA_STATUS_SUCCESS && Descriptor.queue) {
      if (State->NextQueueDestroy(Descriptor.queue) != HSA_STATUS_SUCCESS)
        refuse("cannot destroy a partially configured AMD queue");
      {
        std::lock_guard<std::mutex> Lock(State->Mutex);
        State->ProtectedQueues.erase(Descriptor.queue);
      }
      Descriptor.queue = nullptr;
    }
    if (Status != HSA_STATUS_SUCCESS && FirstError == HSA_STATUS_SUCCESS)
      FirstError = Status;
  }
  return FirstError;
}

hsa_status_t HSA_API toolQueueDestroy(hsa_queue_t *Queue) {
  const hsa_status_t Status = State->NextQueueDestroy(Queue);
  if (Status == HSA_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    State->ProtectedQueues.erase(Queue);
  }
  return Status;
}

hsa_status_t toolQueueGetInfo(hsa_queue_t *Queue,
                              hsa_queue_info_attribute_t Attribute,
                              void *Value) {
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (Attribute == HSA_AMD_QUEUE_INFO_DOORBELL_ID &&
        State->ProtectedQueues.count(Queue))
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  return State->NextQueueGetInfo(Queue, Attribute, Value);
}

bool beginNativeMutation(hsa_executable_t Executable) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  if (State->Executables.find(Executable.handle) != State->Executables.end())
    return false;
  NativeExecutableRecord &Record = State->NativeExecutables[Executable.handle];
  if (Record.SourceRegistrationActive || Record.FreezeActive ||
      Record.Destroying)
    return false;
  ++Record.ActiveMutations;
  return true;
}

void finishNativeMutation(hsa_executable_t Executable, hsa_status_t Status) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->NativeExecutables.find(Executable.handle);
  if (It == State->NativeExecutables.end() || It->second.ActiveMutations == 0)
    refuse("native executable mutation bookkeeping was lost");
  --It->second.ActiveMutations;
  if (Status == HSA_STATUS_SUCCESS)
    It->second.HasContent = true;
  if (It->second.ActiveMutations == 0 && !It->second.HasContent &&
      !It->second.SourceRegistrationActive && !It->second.FreezeActive &&
      !It->second.Destroying)
    State->NativeExecutables.erase(It);
}

bool beginSourceRegistration(hsa_executable_t Executable) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  if (State->Executables.find(Executable.handle) != State->Executables.end())
    return false;
  NativeExecutableRecord &Record = State->NativeExecutables[Executable.handle];
  if (Record.ActiveMutations != 0 || Record.HasContent ||
      Record.SourceRegistrationActive || Record.FreezeActive ||
      Record.Destroying)
    return false;
  Record.ActiveMutations = 1;
  Record.SourceRegistrationActive = true;
  return true;
}

void cancelSourceRegistration(hsa_executable_t Executable) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->NativeExecutables.find(Executable.handle);
  if (It == State->NativeExecutables.end() ||
      !It->second.SourceRegistrationActive || It->second.ActiveMutations != 1 ||
      It->second.HasContent || It->second.FreezeActive || It->second.Destroying)
    refuse("source registration bookkeeping was lost");
  State->NativeExecutables.erase(It);
}

bool commitSourceRegistration(hsa_executable_t Executable,
                              const std::shared_ptr<SourceObject> &Object) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->NativeExecutables.find(Executable.handle);
  if (It == State->NativeExecutables.end() ||
      !It->second.SourceRegistrationActive || It->second.ActiveMutations != 1 ||
      It->second.HasContent || It->second.FreezeActive ||
      It->second.Destroying ||
      State->Executables.find(Executable.handle) != State->Executables.end())
    refuse("source registration reservation was violated");
  for (const std::shared_ptr<KernelRecord> &Kernel : Object->Kernels) {
    const uint64_t SymbolHandle =
        reinterpret_cast<uint64_t>(Kernel->SymbolToken.get());
    const uint64_t KernelHandle =
        reinterpret_cast<uint64_t>(Kernel->KernelToken.get());
    if (!SymbolHandle || !KernelHandle ||
        !State->SymbolTokens.try_emplace(SymbolHandle, Kernel).second ||
        !State->KernelTokens.try_emplace(KernelHandle, Kernel).second)
      refuse("virtual kernel identity collided during source registration");
  }
  State->NativeExecutables.erase(It);
  if (!State->Executables
           .try_emplace(Executable.handle, ExecutableRecord{Object, false})
           .second)
    refuse("source executable handle collided during registration");
  return true;
}

hsa_status_t HSA_API toolLoadCodeObject(hsa_executable_t Executable,
                                        hsa_agent_t Agent,
                                        hsa_code_object_t CodeObject,
                                        const char *Options) {
  if (getAgentView(Agent).NeedsTranslation)
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  if (!beginNativeMutation(Executable))
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  const hsa_status_t Status =
      State->NextLoadCodeObject(Executable, Agent, CodeObject, Options);
  finishNativeMutation(Executable, Status);
  return Status;
}

hsa_status_t HSA_API toolDefineGlobal(hsa_executable_t Executable,
                                      const char *Name, void *Address) {
  if (!beginNativeMutation(Executable))
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  const hsa_status_t Status =
      State->NextDefineGlobal(Executable, Name, Address);
  finishNativeMutation(Executable, Status);
  return Status;
}

hsa_status_t HSA_API toolDefineAgentGlobal(hsa_executable_t Executable,
                                           hsa_agent_t Agent, const char *Name,
                                           void *Address) {
  if (!beginNativeMutation(Executable))
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  const hsa_status_t Status =
      State->NextDefineAgentGlobal(Executable, Agent, Name, Address);
  finishNativeMutation(Executable, Status);
  return Status;
}

hsa_status_t HSA_API toolDefineReadonly(hsa_executable_t Executable,
                                        hsa_agent_t Agent, const char *Name,
                                        void *Address) {
  if (!beginNativeMutation(Executable))
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  const hsa_status_t Status =
      State->NextDefineReadonly(Executable, Agent, Name, Address);
  finishNativeMutation(Executable, Status);
  return Status;
}

} // namespace
} // namespace COMGR::hotswap::hsa_tool

namespace COMGR::hotswap::hsa_tool {
namespace {

bool environmentFlag(const char *Name) {
  const char *Value = std::getenv(Name);
  return Value && *Value && std::strcmp(Value, "0") != 0;
}

bool saveApiTable(HsaApiTable *Table) {
  if (!Table || !Table->core_ || !Table->amd_ext_) {
    llvm::errs() << "hotswap-hsa-tool: HSA runtime supplied an incomplete "
                    "root API table\n";
    return false;
  }
  constexpr size_t CoreEnd =
      offsetof(CoreApiTable, hsa_executable_iterate_program_symbols_fn) +
      sizeof(decltype(CoreApiTable::hsa_executable_iterate_program_symbols_fn));
  constexpr size_t AmdEnd =
      offsetof(AmdExtTable, hsa_amd_queue_create_fn) +
      sizeof(decltype(AmdExtTable::hsa_amd_queue_create_fn));
  if (Table->core_->version.minor_id < CoreEnd ||
      Table->amd_ext_->version.minor_id < AmdEnd) {
    llvm::errs() << "hotswap-hsa-tool: HSA runtime API tables are too small "
                    "for the required interception slots\n";
    return false;
  }

  State->Core = Table->core_;
  State->Amd = Table->amd_ext_;
  CoreApiTable *Core = State->Core;
  AmdExtTable *Amd = State->Amd;

  State->NextIterateAgents = Core->hsa_iterate_agents_fn;
  State->NextAgentGetInfo = Core->hsa_agent_get_info_fn;
  State->NextIsaFromName = Core->hsa_isa_from_name_fn;
  State->NextIsaCompatible = Core->hsa_isa_compatible_fn;
  State->NextAgentIterateIsas = Core->hsa_agent_iterate_isas_fn;
  State->NextIsaGetInfoAlt = Core->hsa_isa_get_info_alt_fn;
  State->NextIsaIterateWavefronts = Core->hsa_isa_iterate_wavefronts_fn;
  State->NextWavefrontGetInfo = Core->hsa_wavefront_get_info_fn;
  State->NextGetExtensionTable = Core->hsa_system_get_extension_table_fn;
  State->NextGetMajorExtensionTable =
      Core->hsa_system_get_major_extension_table_fn;
  State->NextQueueCreate = Core->hsa_queue_create_fn;
  State->NextSoftQueueCreate = Core->hsa_soft_queue_create_fn;
  State->NextQueueDestroy = Core->hsa_queue_destroy_fn;
  State->NextReaderMemory = Core->hsa_code_object_reader_create_from_memory_fn;
  State->NextReaderFile = Core->hsa_code_object_reader_create_from_file_fn;
  State->NextReaderDestroy = Core->hsa_code_object_reader_destroy_fn;
  State->NextExecutableCreate = Core->hsa_executable_create_alt_fn;
  State->NextExecutableDestroy = Core->hsa_executable_destroy_fn;
  State->NextExecutableFreeze = Core->hsa_executable_freeze_fn;
  State->NextExecutableGetInfo = Core->hsa_executable_get_info_fn;
  State->NextExecutableValidate = Core->hsa_executable_validate_fn;
  State->NextExecutableValidateAlt = Core->hsa_executable_validate_alt_fn;
  State->NextLoadProgram = Core->hsa_executable_load_program_code_object_fn;
  State->NextLoadAgent = Core->hsa_executable_load_agent_code_object_fn;
  State->NextLoadCodeObject = Core->hsa_executable_load_code_object_fn;
  State->NextDefineGlobal = Core->hsa_executable_global_variable_define_fn;
  State->NextDefineAgentGlobal =
      Core->hsa_executable_agent_global_variable_define_fn;
  State->NextDefineReadonly = Core->hsa_executable_readonly_variable_define_fn;
  State->NextGetSymbolByName = Core->hsa_executable_get_symbol_by_name_fn;
  State->NextGetSymbol = Core->hsa_executable_get_symbol_fn;
  State->NextSymbolGetInfo = Core->hsa_executable_symbol_get_info_fn;
  State->NextIterateSymbols = Core->hsa_executable_iterate_symbols_fn;
  State->NextIterateAgentSymbols =
      Core->hsa_executable_iterate_agent_symbols_fn;
  State->NextIterateProgramSymbols =
      Core->hsa_executable_iterate_program_symbols_fn;

  State->NextInterceptCreate = Amd->hsa_amd_queue_intercept_create_fn;
  State->NextInterceptRegister = Amd->hsa_amd_queue_intercept_register_fn;
  State->NextQueueGetInfo = Amd->hsa_amd_queue_get_info_fn;
  State->NextAmdQueueCreate = Amd->hsa_amd_queue_create_fn;
  State->NextQueueSetPriority = Amd->hsa_amd_queue_set_priority_fn;
  State->NextQueueSetCuMask = Amd->hsa_amd_queue_cu_set_mask_fn;

  const bool Complete =
      State->NextIterateAgents && State->NextAgentGetInfo &&
      State->NextIsaFromName && State->NextIsaCompatible &&
      State->NextAgentIterateIsas && State->NextIsaGetInfoAlt &&
      State->NextIsaIterateWavefronts && State->NextWavefrontGetInfo &&
      State->NextGetExtensionTable && State->NextGetMajorExtensionTable &&
      State->NextQueueCreate && State->NextSoftQueueCreate &&
      State->NextQueueDestroy && State->NextReaderMemory &&
      State->NextReaderFile && State->NextReaderDestroy &&
      State->NextExecutableCreate && State->NextExecutableDestroy &&
      State->NextExecutableFreeze && State->NextExecutableGetInfo &&
      State->NextExecutableValidate && State->NextExecutableValidateAlt &&
      State->NextLoadProgram && State->NextLoadAgent &&
      State->NextLoadCodeObject && State->NextDefineGlobal &&
      State->NextDefineAgentGlobal && State->NextDefineReadonly &&
      State->NextGetSymbolByName && State->NextGetSymbol &&
      State->NextSymbolGetInfo && State->NextIterateSymbols &&
      State->NextIterateAgentSymbols && State->NextIterateProgramSymbols &&
      State->NextInterceptCreate && State->NextInterceptRegister &&
      State->NextQueueGetInfo && State->NextAmdQueueCreate &&
      State->NextQueueSetPriority && State->NextQueueSetCuMask;
  if (!Complete)
    llvm::errs() << "hotswap-hsa-tool: HSA runtime API tables omit a required "
                    "interception entry\n";
  return Complete;
}

bool configure() {
  const char *Presented = std::getenv("HSA_HOTSWAP_PRESENT_ISA");
  if (!Presented || !*Presented) {
    State->Active = false;
    return true;
  }

  std::string Requested = Presented;
  if (Requested.rfind("amdgcn-amd-amdhsa--", 0) != 0)
    Requested = "amdgcn-amd-amdhsa--" + Requested;
  if (State->NextIsaFromName(Requested.c_str(), &State->PresentedIsa) !=
          HSA_STATUS_SUCCESS ||
      !State->PresentedIsa.handle) {
    llvm::errs() << "hotswap-hsa-tool: invalid presented ISA " << Presented
                 << '\n';
    return false;
  }
  State->PresentedName = isaName(State->PresentedIsa);
  State->PresentedGfx = processor(State->PresentedName);
  if (State->PresentedName.empty() || State->PresentedGfx.empty() ||
      State->PresentedGfx.size() >= AgentNameSize) {
    llvm::errs() << "hotswap-hsa-tool: cannot resolve the canonical name for "
                    "the presented ISA or represent it as an HSA agent name\n";
    return false;
  }

  if (const char *Value = std::getenv("HSA_HOTSWAP_CACHE_DIR"))
    State->CacheDirectory = Value;
  if (const char *Value = std::getenv("HSA_HOTSWAP_PROOF_LOG"))
    State->ProofPath = Value;
  State->AssumeHipGlobalOffsetZero =
      environmentFlag("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO");
  State->Active = true;
  return true;
}

void install() {
  CoreApiTable *Core = State->Core;
  AmdExtTable *Amd = State->Amd;
  Core->hsa_agent_get_info_fn = toolAgentGetInfo;
  Core->hsa_agent_iterate_isas_fn = toolAgentIterateIsas;
  Core->hsa_system_get_extension_table_fn = toolGetExtensionTable;
  Core->hsa_system_get_major_extension_table_fn = toolGetMajorExtensionTable;
  Core->hsa_queue_create_fn = toolQueueCreate;
  Core->hsa_soft_queue_create_fn = toolSoftQueueCreate;
  Core->hsa_queue_destroy_fn = toolQueueDestroy;
  Core->hsa_code_object_reader_create_from_memory_fn = toolReaderMemory;
  Core->hsa_code_object_reader_create_from_file_fn = toolReaderFile;
  Core->hsa_code_object_reader_destroy_fn = toolReaderDestroy;
  Core->hsa_executable_destroy_fn = toolExecutableDestroy;
  Core->hsa_executable_freeze_fn = toolExecutableFreeze;
  Core->hsa_executable_validate_fn = toolExecutableValidate;
  Core->hsa_executable_validate_alt_fn = toolExecutableValidateAlt;
  Core->hsa_executable_load_program_code_object_fn = toolLoadProgram;
  Core->hsa_executable_load_agent_code_object_fn = toolLoadAgent;
  Core->hsa_executable_load_code_object_fn = toolLoadCodeObject;
  Core->hsa_executable_global_variable_define_fn = toolDefineGlobal;
  Core->hsa_executable_agent_global_variable_define_fn = toolDefineAgentGlobal;
  Core->hsa_executable_readonly_variable_define_fn = toolDefineReadonly;
  Core->hsa_executable_get_symbol_by_name_fn = toolGetSymbolByName;
  Core->hsa_executable_get_symbol_fn = toolGetSymbol;
  Core->hsa_executable_symbol_get_info_fn = toolSymbolGetInfo;
  Core->hsa_executable_iterate_symbols_fn = toolIterateSymbols;
  Core->hsa_executable_iterate_agent_symbols_fn = toolIterateAgentSymbols;
  Core->hsa_executable_iterate_program_symbols_fn = toolIterateProgramSymbols;
  Amd->hsa_amd_queue_intercept_create_fn = toolInterceptCreate;
  Amd->hsa_amd_queue_get_info_fn = toolQueueGetInfo;
  Amd->hsa_amd_queue_create_fn = toolAmdQueueCreate;
  State->Installed = true;
}

template <typename FunctionT> struct FunctionRestore {
  FunctionT *Wrapper = nullptr;
  FunctionT *Saved = nullptr;
};

template <typename FunctionT>
void restoreFunction(FunctionT *&Slot, FunctionRestore<FunctionT> Restore) {
  if (Slot == Restore.Wrapper) {
    Slot = Restore.Saved;
    return;
  }

  // A tool installed later may still own this slot. Clobbering its wrapper
  // would break API-tool chaining, while restoring through an opaque function
  // pointer is impossible. ROCr resets the complete table immediately after
  // reverse-order OnUnload callbacks; leave the outer wrapper in place until
  // that reset and restore every slot that is still directly ours.
  State->ApiRestoreDeferred = true;
}

void restore() {
  if (!State || !State->Installed)
    return;
  CoreApiTable *Core = State->Core;
  AmdExtTable *Amd = State->Amd;
  restoreFunction(Core->hsa_agent_get_info_fn,
                  {toolAgentGetInfo, State->NextAgentGetInfo});
  restoreFunction(Core->hsa_agent_iterate_isas_fn,
                  {toolAgentIterateIsas, State->NextAgentIterateIsas});
  restoreFunction(Core->hsa_system_get_extension_table_fn,
                  {toolGetExtensionTable, State->NextGetExtensionTable});
  restoreFunction(
      Core->hsa_system_get_major_extension_table_fn,
      {toolGetMajorExtensionTable, State->NextGetMajorExtensionTable});
  restoreFunction(Core->hsa_queue_create_fn,
                  {toolQueueCreate, State->NextQueueCreate});
  restoreFunction(Core->hsa_soft_queue_create_fn,
                  {toolSoftQueueCreate, State->NextSoftQueueCreate});
  restoreFunction(Core->hsa_queue_destroy_fn,
                  {toolQueueDestroy, State->NextQueueDestroy});
  restoreFunction(Core->hsa_code_object_reader_create_from_memory_fn,
                  {toolReaderMemory, State->NextReaderMemory});
  restoreFunction(Core->hsa_code_object_reader_create_from_file_fn,
                  {toolReaderFile, State->NextReaderFile});
  restoreFunction(Core->hsa_code_object_reader_destroy_fn,
                  {toolReaderDestroy, State->NextReaderDestroy});
  restoreFunction(Core->hsa_executable_destroy_fn,
                  {toolExecutableDestroy, State->NextExecutableDestroy});
  restoreFunction(Core->hsa_executable_freeze_fn,
                  {toolExecutableFreeze, State->NextExecutableFreeze});
  restoreFunction(Core->hsa_executable_validate_fn,
                  {toolExecutableValidate, State->NextExecutableValidate});
  restoreFunction(
      Core->hsa_executable_validate_alt_fn,
      {toolExecutableValidateAlt, State->NextExecutableValidateAlt});
  restoreFunction(Core->hsa_executable_load_program_code_object_fn,
                  {toolLoadProgram, State->NextLoadProgram});
  restoreFunction(Core->hsa_executable_load_agent_code_object_fn,
                  {toolLoadAgent, State->NextLoadAgent});
  restoreFunction(Core->hsa_executable_load_code_object_fn,
                  {toolLoadCodeObject, State->NextLoadCodeObject});
  restoreFunction(Core->hsa_executable_global_variable_define_fn,
                  {toolDefineGlobal, State->NextDefineGlobal});
  restoreFunction(Core->hsa_executable_agent_global_variable_define_fn,
                  {toolDefineAgentGlobal, State->NextDefineAgentGlobal});
  restoreFunction(Core->hsa_executable_readonly_variable_define_fn,
                  {toolDefineReadonly, State->NextDefineReadonly});
  restoreFunction(Core->hsa_executable_get_symbol_by_name_fn,
                  {toolGetSymbolByName, State->NextGetSymbolByName});
  restoreFunction(Core->hsa_executable_get_symbol_fn,
                  {toolGetSymbol, State->NextGetSymbol});
  restoreFunction(Core->hsa_executable_symbol_get_info_fn,
                  {toolSymbolGetInfo, State->NextSymbolGetInfo});
  restoreFunction(Core->hsa_executable_iterate_symbols_fn,
                  {toolIterateSymbols, State->NextIterateSymbols});
  restoreFunction(Core->hsa_executable_iterate_agent_symbols_fn,
                  {toolIterateAgentSymbols, State->NextIterateAgentSymbols});
  restoreFunction(
      Core->hsa_executable_iterate_program_symbols_fn,
      {toolIterateProgramSymbols, State->NextIterateProgramSymbols});
  restoreFunction(Amd->hsa_amd_queue_intercept_create_fn,
                  {toolInterceptCreate, State->NextInterceptCreate});
  restoreFunction(Amd->hsa_amd_queue_get_info_fn,
                  {toolQueueGetInfo, State->NextQueueGetInfo});
  restoreFunction(Amd->hsa_amd_queue_create_fn,
                  {toolAmdQueueCreate, State->NextAmdQueueCreate});
  State->Installed = false;
}

uint64_t releaseRemainingChildrenForRuntimeTeardown() {
  std::unique_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  std::vector<std::shared_ptr<SourceObject>> Objects;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    for (const auto &Entry : State->NativeExecutables)
      if (Entry.second.ActiveMutations != 0 ||
          Entry.second.SourceRegistrationActive || Entry.second.Destroying)
        refuse("tool unload raced an executable mutation");
    Objects.reserve(State->Executables.size());
    for (auto &Entry : State->Executables) {
      const std::shared_ptr<SourceObject> &Object = Entry.second.Object;
      {
        std::lock_guard<std::mutex> LifetimeLock(Object->LifetimeMutex);
        if (Object->ActiveIterations != 0)
          refuse("tool unload raced an active source symbol iteration");
        Object->Alive.store(false, std::memory_order_release);
        Object->RetirementRequested = true;
        Object->CleanupStarted = true;
      }
      Objects.push_back(Object);
    }
    State->Executables.clear();
    State->SymbolTokens.clear();
    State->KernelTokens.clear();
  }
  uint64_t ChildCount = 0;
  for (const std::shared_ptr<SourceObject> &Object : Objects) {
    for (const std::shared_ptr<KernelRecord> &Kernel : Object->Kernels) {
      std::lock_guard<std::mutex> Lock(Kernel->Mutex);
      if (Kernel->Child.handle) {
        ++ChildCount;
        Kernel->Child = {};
      }
      Kernel->SymbolToken.reset();
      Kernel->KernelToken.reset();
      Kernel->Object.reset();
    }
  }
  return ChildCount;
}

std::string summary() {
  const uint64_t Intercepted =
      State->Count.InterceptedDispatches.load(std::memory_order_relaxed);
  const uint64_t Rewritten =
      State->Count.RewrittenDispatches.load(std::memory_order_relaxed);
  return "\"event\":\"coverage_summary\",\"registered_source_objects\":" +
         std::to_string(
             State->Count.RegisteredObjects.load(std::memory_order_relaxed)) +
         ",\"registered_kernels\":" +
         std::to_string(
             State->Count.RegisteredKernels.load(std::memory_order_relaxed)) +
         ",\"translation_requests\":" +
         std::to_string(
             State->Count.TranslationRequests.load(std::memory_order_relaxed)) +
         ",\"successful_translations\":" +
         std::to_string(State->Count.SuccessfulTranslations.load(
             std::memory_order_relaxed)) +
         ",\"failed_translations\":" +
         std::to_string(
             State->Count.FailedTranslations.load(std::memory_order_relaxed)) +
         ",\"cache_hits\":" +
         std::to_string(
             State->Count.CacheHits.load(std::memory_order_relaxed)) +
         ",\"intercepted_dispatches\":" + std::to_string(Intercepted) +
         ",\"rewritten_dispatches\":" + std::to_string(Rewritten) +
         ",\"protected_queues_created\":" +
         std::to_string(
             State->Count.ProtectedQueues.load(std::memory_order_relaxed)) +
         ",\"rejected_source_objects\":" +
         std::to_string(
             State->Count.RejectedObjects.load(std::memory_order_relaxed)) +
         ",\"all_intercepted_dispatches_rewritten\":" +
         (Intercepted == Rewritten ? "true" : "false");
}

} // namespace
} // namespace COMGR::hotswap::hsa_tool

extern "C" {

HSA_HOTSWAP_EXPORT bool OnLoad(HsaApiTable *Table, uint64_t RuntimeVersion,
                               uint64_t, const char *const *) {
  using namespace COMGR::hotswap::hsa_tool;
  if (State) {
    llvm::errs() << "hotswap-hsa-tool: refusing duplicate tool load\n";
    return false;
  }
  if (RuntimeVersion != HSA_API_TABLE_MAJOR_VERSION) {
    llvm::errs() << "hotswap-hsa-tool: incompatible HSA API-table major "
                    "version "
                 << RuntimeVersion << "; expected "
                 << HSA_API_TABLE_MAJOR_VERSION << '\n';
    return false;
  }
  const char *Presented = std::getenv("HSA_HOTSWAP_PRESENT_ISA");
  if (!Presented || !*Presented)
    return true;
  State = new (std::nothrow) ToolState;
  if (!State) {
    llvm::errs() << "hotswap-hsa-tool: cannot allocate tool state\n";
    return false;
  }
  if (!saveApiTable(Table) || !configure()) {
    delete State;
    State = nullptr;
    return false;
  }
  const hsa_status_t Discovery =
      State->NextIterateAgents(discoverAgent, nullptr);
  if (Discovery != HSA_STATUS_SUCCESS || State->Agents.empty()) {
    llvm::errs() << "hotswap-hsa-tool: GPU discovery failed with HSA status "
                 << static_cast<int>(Discovery) << "; discovered "
                 << State->Agents.size() << " compatible GPU agents\n";
    delete State;
    State = nullptr;
    return false;
  }
  install();
  if (!writeProof("\"event\":\"tool_loaded\",\"presented_isa\":\"" +
                  jsonEscape(State->PresentedName) + "\"")) {
    restore();
    delete State;
    State = nullptr;
    return false;
  }
  llvm::errs() << "hotswap-hsa-tool: presenting " << State->PresentedName
               << '\n';
  return true;
}

HSA_HOTSWAP_EXPORT void OnUnload() {
  using namespace COMGR::hotswap::hsa_tool;
  if (!State)
    return;
  if (!State->Active) {
    delete State;
    State = nullptr;
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    if (!State->ProtectedQueues.empty())
      refuse("tool unload encountered live protected queues; destroy every "
             "queue before final HSA shutdown");
  }

  restore();
  if (State->ApiRestoreDeferred) {
    proofOrRefuse("\"event\":\"api_table_restore_deferred\","
                  "\"reason\":\"a later wrapper still owns one or more "
                  "slots\"");
  }
  const std::string Coverage = summary();
  if (!writeProof(Coverage))
    refuse("configured proof log could not be written at unload");
  const uint64_t Intercepted =
      State->Count.InterceptedDispatches.load(std::memory_order_relaxed);
  const uint64_t Rewritten =
      State->Count.RewrittenDispatches.load(std::memory_order_relaxed);
  if (Intercepted != Rewritten)
    refuse("coverage invariant failed at unload");
  // ROCr invokes tool unload callbacks after its public HSA reference count
  // reaches zero. Public executable-destruction calls therefore return
  // HSA_STATUS_ERROR_NOT_INITIALIZED here. The loader remains alive until
  // immediately after tool unload and owns every executable created through
  // the public API, so release our host-side references and let that loader
  // teardown destroy any child whose parent was not explicitly destroyed.
  const uint64_t RuntimeChildren = releaseRemainingChildrenForRuntimeTeardown();
  proofOrRefuse("\"event\":\"runtime_teardown_children\",\"count\":" +
                std::to_string(RuntimeChildren));
  proofOrRefuse("\"event\":\"tool_unloaded\"");
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    State->Readers.clear();
    State->Agents.clear();
    State->ProtectedQueues.clear();
  }
  delete State;
  State = nullptr;
}

} // extern "C"

namespace COMGR::hotswap::hsa_tool {
namespace {

void destroyKernelChild(KernelRecord &Kernel) {
  if (Kernel.Child.handle) {
    const hsa_status_t Status = State->NextExecutableDestroy(Kernel.Child);
    if (Status != HSA_STATUS_SUCCESS)
      refuse("cannot destroy translated child executable for " +
             Kernel.MetadataName + ": HSA status " +
             std::to_string(static_cast<int>(Status)));
    Kernel.Child = {};
  }
}

void cleanupSourceObject(const std::shared_ptr<SourceObject> &Object) {
  std::unique_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  for (const std::shared_ptr<KernelRecord> &Kernel : Object->Kernels) {
    std::lock_guard<std::mutex> KernelLock(Kernel->Mutex);
    {
      std::lock_guard<std::mutex> Lock(State->Mutex);
      const uint64_t SymbolHandle =
          reinterpret_cast<uint64_t>(Kernel->SymbolToken.get());
      auto Symbol = State->SymbolTokens.find(SymbolHandle);
      if (Symbol == State->SymbolTokens.end() || Symbol->second != Kernel)
        refuse("virtual symbol bookkeeping was lost during destruction");
      Symbol->second.reset();
      State->KernelTokens.erase(
          reinterpret_cast<uint64_t>(Kernel->KernelToken.get()));
      if (Kernel->SymbolToken)
        State->RetiredSymbolTokens.push_back(std::move(Kernel->SymbolToken));
      if (Kernel->KernelToken)
        State->RetiredKernelTokens.push_back(std::move(Kernel->KernelToken));
    }
    destroyKernelChild(*Kernel);
    Kernel->Object.reset();
  }
}

bool requestSourceObjectCleanup(const std::shared_ptr<SourceObject> &Object) {
  std::lock_guard<std::mutex> Lock(Object->LifetimeMutex);
  Object->Alive.store(false, std::memory_order_release);
  Object->RetirementRequested = true;
  if (Object->ActiveIterations != 0 || Object->CleanupStarted)
    return false;
  Object->CleanupStarted = true;
  return true;
}

class SourceIterationLease {
public:
  static std::unique_ptr<SourceIterationLease>
  acquire(const std::shared_ptr<SourceObject> &Object) {
    std::lock_guard<std::mutex> Lock(Object->LifetimeMutex);
    if (Object->RetirementRequested)
      return {};
    ++Object->ActiveIterations;
    std::unique_ptr<SourceIterationLease> Lease(
        new (std::nothrow) SourceIterationLease(Object));
    if (!Lease)
      --Object->ActiveIterations;
    return Lease;
  }

  ~SourceIterationLease() {
    bool Cleanup = false;
    {
      std::lock_guard<std::mutex> Lock(Object->LifetimeMutex);
      if (Object->ActiveIterations == 0)
        refuse("source symbol iteration bookkeeping underflowed");
      --Object->ActiveIterations;
      if (Object->ActiveIterations == 0 && Object->RetirementRequested &&
          !Object->CleanupStarted) {
        Object->CleanupStarted = true;
        Cleanup = true;
      }
    }
    if (Cleanup)
      cleanupSourceObject(Object);
  }

private:
  explicit SourceIterationLease(std::shared_ptr<SourceObject> Object)
      : Object(std::move(Object)) {}

  std::shared_ptr<SourceObject> Object;
};

bool translateKernel(KernelRecord &Kernel) {
  const std::shared_ptr<SourceObject> ObjectOwner = Kernel.Object.lock();
  if (!ObjectOwner) {
    Kernel.Failure = "parent executable no longer exists";
    return false;
  }
  SourceObject &Object = *ObjectOwner;
  if (!Object.Alive.load(std::memory_order_acquire)) {
    Kernel.Failure = "parent executable is being destroyed";
    return false;
  }

  State->Count.TranslationRequests.fetch_add(1, std::memory_order_relaxed);
  proofOrRefuse("\"event\":\"translation_requested\",\"kernel\":\"" +
                jsonEscape(Kernel.MetadataName) + "\",\"source_isa\":\"" +
                jsonEscape(Object.SourceIsa) + "\",\"target_isa\":\"" +
                jsonEscape(Object.TargetIsa) + "\"");

  DataOwner Input;
  DataOwner Output;
  TranspileResultOwner Result;
  if (!createComgrData(Object.SourceElf, Input)) {
    Kernel.Failure = "cannot create COMGR translation input";
    return false;
  }

  amd_comgr_hotswap_transpile_options_v2_t Options{};
  Options.version = AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_VERSION_2;
  Options.cache_directory =
      State->CacheDirectory.empty() ? nullptr : State->CacheDirectory.c_str();
  Options.kernel_name = Kernel.MetadataName.c_str();
  Options.flags = AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME |
                  AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_STRICT;
  if (State->AssumeHipGlobalOffsetZero)
    Options.flags |=
        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_ASSUME_HIP_GLOBAL_OFFSET_ZERO;

  const amd_comgr_status_t ComgrStatus =
      amd_comgr_hotswap_transpile_with_options_v2(
          Input.Data, Object.SourceIsa.c_str(), Object.TargetIsa.c_str(),
          &Options, &Output.Data, &Result.Result);
  if (ComgrStatus != AMD_COMGR_STATUS_SUCCESS) {
    Kernel.Failure = resultString(
        Result.Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL);
    if (Kernel.Failure.empty())
      Kernel.Failure = "COMGR per-kernel translation failed";
    return false;
  }

  int64_t Scale = 0;
  bool CacheHit = false;
  if (!Result.Result.handle) {
    Kernel.Failure = "COMGR returned no translation result metadata";
    return false;
  }
  if (amd_comgr_hotswap_transpile_result_get_info(
          Result.Result,
          AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SCALED_DISPATCH_FACTOR,
          &Scale) != AMD_COMGR_STATUS_SUCCESS) {
    Kernel.Failure = "COMGR did not return a dispatch scale";
    return false;
  }
  if (amd_comgr_hotswap_transpile_result_get_info(
          Result.Result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT,
          &CacheHit) != AMD_COMGR_STATUS_SUCCESS) {
    Kernel.Failure = "COMGR did not return cache-result metadata";
    return false;
  }
  if (Scale < 1 ||
      Scale > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    Kernel.Failure =
        "COMGR returned invalid dispatch scale " + std::to_string(Scale);
    return false;
  }

  size_t OutputSize = 0;
  if (!Output.Data.handle ||
      amd_comgr_get_data(Output.Data, &OutputSize, nullptr) !=
          AMD_COMGR_STATUS_SUCCESS ||
      OutputSize == 0) {
    Kernel.Failure = "COMGR returned an empty translated object";
    return false;
  }
  std::unique_ptr<uint8_t[]> Target(new (std::nothrow) uint8_t[OutputSize]);
  if (!Target) {
    Kernel.Failure = "cannot allocate translated object storage";
    return false;
  }
  const size_t TargetCapacity = OutputSize;
  if (amd_comgr_get_data(Output.Data, &OutputSize,
                         reinterpret_cast<char *>(Target.get())) !=
          AMD_COMGR_STATUS_SUCCESS ||
      OutputSize == 0 || OutputSize > TargetCapacity) {
    Kernel.Failure = "cannot copy the translated object from COMGR";
    return false;
  }

  hsa_executable_t Child{};
  hsa_code_object_reader_t Reader{};
  hsa_loaded_code_object_t Loaded{};
  hsa_executable_symbol_t Symbol{};
  const char *LoadStage = "creating translated child executable";
  hsa_status_t Status = State->NextExecutableCreate(
      Object.Profile, Object.Rounding, nullptr, &Child);
  if (Status == HSA_STATUS_SUCCESS && !Child.handle) {
    Kernel.Failure = "runtime created a null translated child executable";
    Status = HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "creating translated code-object reader";
    Status = State->NextReaderMemory(Target.get(), OutputSize, &Reader);
    if (Status == HSA_STATUS_SUCCESS && !Reader.handle) {
      Kernel.Failure = "runtime created a null translated object reader";
      Status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;
    }
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "loading translated code object";
    Status = State->NextLoadAgent(
        Child, Object.Agent, Reader,
        Object.Options.empty() ? nullptr : Object.Options.c_str(), &Loaded);
    if (Status == HSA_STATUS_SUCCESS && !Loaded.handle) {
      Kernel.Failure = "runtime returned a null translated loaded-code object";
      Status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "destroying translated code-object reader";
    Status = State->NextReaderDestroy(Reader);
    if (Status == HSA_STATUS_SUCCESS)
      Reader = {};
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "freezing translated child executable";
    Status = State->NextExecutableFreeze(Child, nullptr);
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "resolving translated kernel symbol";
    Status = State->NextGetSymbolByName(Child, Kernel.SymbolName.c_str(),
                                        &Object.Agent, &Symbol);
    if (Status == HSA_STATUS_SUCCESS && !Symbol.handle) {
      Kernel.Failure = "translated kernel resolved to a null symbol";
      Status = HSA_STATUS_ERROR_INVALID_SYMBOL_NAME;
    }
  }

  KernelDispatchTarget TargetInfo;
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "querying translated kernel object";
    Status = State->NextSymbolGetInfo(Symbol,
                                      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                      &TargetInfo.KernelObject);
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "querying translated private-segment size";
    Status = State->NextSymbolGetInfo(
        Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
        &TargetInfo.TargetPrivateSegmentSize);
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "querying translated group-segment size";
    Status = State->NextSymbolGetInfo(
        Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
        &TargetInfo.TargetGroupSegmentSize);
  }
  uint32_t TargetKernargSize = 0;
  uint32_t TargetKernargAlignment = 0;
  bool TargetDynamicCallstack = false;
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "querying translated kernarg size";
    Status = State->NextSymbolGetInfo(
        Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
        &TargetKernargSize);
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "querying translated kernarg alignment";
    Status = State->NextSymbolGetInfo(
        Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
        &TargetKernargAlignment);
  }
  if (Status == HSA_STATUS_SUCCESS) {
    LoadStage = "querying translated dynamic-callstack state";
    Status = State->NextSymbolGetInfo(
        Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK,
        &TargetDynamicCallstack);
  }
  const uint32_t PresentedKernargAlignment =
      std::max<uint32_t>(16, Kernel.KernargSegmentAlignment);
  if (Status == HSA_STATUS_SUCCESS &&
      (TargetKernargSize != Kernel.KernargSegmentSize ||
       TargetKernargAlignment == 0 ||
       (TargetKernargAlignment & (TargetKernargAlignment - 1)) != 0 ||
       TargetKernargAlignment > PresentedKernargAlignment)) {
    Kernel.Failure =
        "translated kernarg ABI differs from source (source size " +
        std::to_string(Kernel.KernargSegmentSize) + ", target size " +
        std::to_string(TargetKernargSize) + ", source alignment " +
        std::to_string(PresentedKernargAlignment) + ", target alignment " +
        std::to_string(TargetKernargAlignment) + ")";
    Status = HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }
  if (Status == HSA_STATUS_SUCCESS && TargetDynamicCallstack) {
    Kernel.Failure = "translated kernel unexpectedly uses a dynamic call "
                     "stack";
    Status = HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }
  TargetInfo.SourcePrivateSegmentSize = Kernel.SourcePrivateSegmentSize;
  TargetInfo.SourceGroupSegmentSize = Kernel.SourceGroupSegmentSize;
  TargetInfo.Scale = static_cast<uint32_t>(Scale);
  TargetInfo.MaxWorkgroupSizeX = Object.TargetMaxWorkgroupDim[0];
  TargetInfo.MaxWorkgroupSizeY = Object.TargetMaxWorkgroupDim[1];
  TargetInfo.MaxWorkgroupSizeZ = Object.TargetMaxWorkgroupDim[2];
  TargetInfo.MaxWorkgroupSize = Object.TargetMaxWorkgroupSize;
  TargetInfo.MaxGridSizeX = Object.TargetMaxGridDim.x;
  TargetInfo.MaxGridSizeY = Object.TargetMaxGridDim.y;
  TargetInfo.MaxGridSizeZ = Object.TargetMaxGridDim.z;
  TargetInfo.MaxGridSize = Object.TargetMaxGridSize;

  if (Status != HSA_STATUS_SUCCESS || TargetInfo.KernelObject == 0) {
    if (Kernel.Failure.empty()) {
      if (Status != HSA_STATUS_SUCCESS)
        Kernel.Failure = std::string(LoadStage) + " failed with HSA status " +
                         std::to_string(static_cast<int>(Status));
      else if (TargetInfo.KernelObject == 0)
        Kernel.Failure = "translated kernel resolved to a null kernel object";
    }
    if (Child.handle &&
        State->NextExecutableDestroy(Child) != HSA_STATUS_SUCCESS)
      refuse("cannot roll back a failed translated child executable");
    if (Reader.handle && State->NextReaderDestroy(Reader) != HSA_STATUS_SUCCESS)
      refuse("cannot roll back a failed translated object reader");
    return false;
  }

  if (!Object.Alive.load(std::memory_order_acquire)) {
    if (State->NextExecutableDestroy(Child) != HSA_STATUS_SUCCESS)
      refuse("cannot destroy translated child after parent destruction");
    if (Reader.handle && State->NextReaderDestroy(Reader) != HSA_STATUS_SUCCESS)
      refuse("cannot destroy translated reader after parent destruction");
    Kernel.Failure = "parent executable was destroyed during translation";
    return false;
  }

  Kernel.Child = Child;
  Kernel.Symbol = Symbol;
  Kernel.Target = TargetInfo;
  if (!Object.Alive.load(std::memory_order_relaxed)) {
    destroyKernelChild(Kernel);
    Kernel.Failure = "parent executable was destroyed during translation";
    return false;
  }

  if (CacheHit)
    State->Count.CacheHits.fetch_add(1, std::memory_order_relaxed);
  State->Count.SuccessfulTranslations.fetch_add(1, std::memory_order_relaxed);
  proofOrRefuse("\"event\":\"translation_succeeded\",\"kernel\":\"" +
                jsonEscape(Kernel.MetadataName) + "\",\"source_gfx\":\"" +
                jsonEscape(Object.SourceGfx) + "\",\"target_gfx\":\"" +
                jsonEscape(Object.TargetGfx) +
                "\",\"cache_hit\":" + (CacheHit ? "true" : "false") +
                ",\"scale\":" + std::to_string(TargetInfo.Scale));
  return true;
}

bool ensureTranslated(const std::shared_ptr<KernelRecord> &Kernel) {
  std::lock_guard<std::mutex> Lock(Kernel->Mutex);
  if (!Kernel->Attempted) {
    Kernel->Attempted = true;
    Kernel->Succeeded = translateKernel(*Kernel);
    if (!Kernel->Succeeded) {
      State->Count.FailedTranslations.fetch_add(1, std::memory_order_relaxed);
      proofOrRefuse("\"event\":\"translation_failed\",\"kernel\":\"" +
                    jsonEscape(Kernel->MetadataName) + "\",\"reason\":\"" +
                    jsonEscape(Kernel->Failure) + "\"");
    }
  }
  return Kernel->Succeeded;
}

std::shared_ptr<KernelRecord> findSourceKernel(hsa_executable_t Executable,
                                               const char *Name,
                                               const hsa_agent_t *Agent,
                                               bool *Frozen = nullptr) {
  if (!Name)
    return {};
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->Executables.find(Executable.handle);
  if (It == State->Executables.end())
    return {};
  if (Frozen)
    *Frozen = It->second.Frozen;
  const std::shared_ptr<SourceObject> &Object = It->second.Object;
  // Source kernels are agent symbols, never program symbols.
  if (!Agent || Agent->handle != Object->Agent.handle)
    return {};
  for (const std::shared_ptr<KernelRecord> &Kernel : Object->Kernels) {
    if (Kernel->MetadataName == Name || Kernel->SymbolName == Name)
      return Kernel;
  }
  return {};
}

void recordReader(hsa_code_object_reader_t Reader, Bytes Copy) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  if (!Reader.handle ||
      !State->Readers.try_emplace(Reader.handle, std::move(Copy)).second)
    refuse("runtime reused a live code-object reader handle");
}

hsa_status_t HSA_API toolReaderMemory(const void *Object, size_t Size,
                                      hsa_code_object_reader_t *Reader) {
  if (!Object || Size == 0 || !Reader ||
      Size > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  hsa_status_t Status = State->NextReaderMemory(Object, Size, Reader);
  if (Status != HSA_STATUS_SUCCESS)
    return Status;
  Bytes Copy(new (std::nothrow) std::vector<uint8_t>);
  if (!Copy) {
    if (State->NextReaderDestroy(*Reader) != HSA_STATUS_SUCCESS)
      refuse("cannot destroy code-object reader after allocation failure");
    *Reader = {};
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }
  const llvm::ArrayRef<uint8_t> Source(static_cast<const uint8_t *>(Object),
                                       Size);
  Copy->assign(Source.begin(), Source.end());
  recordReader(*Reader, std::move(Copy));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolReaderFile(hsa_file_t File,
                                    hsa_code_object_reader_t *Reader) {
  if (!Reader)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  Bytes Copy = readWholeFile(File);
  if (!Copy)
    return HSA_STATUS_ERROR_INVALID_FILE;
  hsa_status_t Status = State->NextReaderFile(File, Reader);
  if (Status == HSA_STATUS_SUCCESS)
    recordReader(*Reader, std::move(Copy));
  return Status;
}

hsa_status_t HSA_API toolReaderFileSlice(hsa_file_t File, size_t Offset,
                                         size_t Size,
                                         hsa_code_object_reader_t *Reader) {
  if (!Reader || Size == 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  Bytes Copy = readFile(File, Offset, Size);
  if (!Copy)
    return HSA_STATUS_ERROR_INVALID_FILE;
  if (!State->NextLoaderReaderFileSlice)
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  hsa_status_t Status =
      State->NextLoaderReaderFileSlice(File, Offset, Size, Reader);
  if (Status == HSA_STATUS_SUCCESS)
    recordReader(*Reader, std::move(Copy));
  return Status;
}

hsa_status_t HSA_API toolReaderDestroy(hsa_code_object_reader_t Reader) {
  const hsa_status_t Status = State->NextReaderDestroy(Reader);
  if (Status == HSA_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    State->Readers.erase(Reader.handle);
  }
  return Status;
}

Bytes readerBytes(hsa_code_object_reader_t Reader) {
  std::lock_guard<std::mutex> Lock(State->Mutex);
  auto It = State->Readers.find(Reader.handle);
  return It == State->Readers.end() ? Bytes{} : It->second;
}

hsa_status_t rejectSourceObject(hsa_status_t Status,
                                const std::string &Reason) {
  llvm::errs() << "hotswap-hsa-tool: rejecting source object: " << Reason
               << '\n';
  State->Count.RejectedObjects.fetch_add(1, std::memory_order_relaxed);
  proofOrRefuse("\"event\":\"code_object_rejected\",\"reason\":\"" +
                jsonEscape(Reason) + "\"");
  return Status;
}

hsa_status_t HSA_API toolLoadAgent(hsa_executable_t Executable,
                                   hsa_agent_t Agent,
                                   hsa_code_object_reader_t Reader,
                                   const char *Options,
                                   hsa_loaded_code_object_t *Loaded) {
  const AgentView View = getAgentView(Agent);
  if (!View.NeedsTranslation) {
    if (!beginNativeMutation(Executable))
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    const hsa_status_t Status =
        State->NextLoadAgent(Executable, Agent, Reader, Options, Loaded);
    finishNativeMutation(Executable, Status);
    return Status;
  }
  if (Loaded)
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  if (!beginSourceRegistration(Executable))
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  llvm::scope_exit Registration([&] { cancelSourceRegistration(Executable); });

  Bytes Source = readerBytes(Reader);
  if (!Source)
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER;

  hsa_executable_state_t ExecutableState;
  hsa_profile_t Profile;
  hsa_default_float_rounding_mode_t Rounding;
  if (State->NextExecutableGetInfo(Executable, HSA_EXECUTABLE_INFO_STATE,
                                   &ExecutableState) != HSA_STATUS_SUCCESS ||
      State->NextExecutableGetInfo(Executable, HSA_EXECUTABLE_INFO_PROFILE,
                                   &Profile) != HSA_STATUS_SUCCESS ||
      State->NextExecutableGetInfo(
          Executable, HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE,
          &Rounding) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  if (ExecutableState == HSA_EXECUTABLE_STATE_FROZEN)
    return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;

  std::string SourceIsa;
  std::vector<KernelMetadata> Metadata;
  std::string Failure;
  if (!inspectSourceObject(Source, SourceIsa, Metadata, Failure))
    return rejectSourceObject(HSA_STATUS_ERROR_INVALID_CODE_OBJECT, Failure);
  if (processor(SourceIsa) != View.PresentedGfx) {
    Failure = "source ISA " + SourceIsa + " does not match presented ISA " +
              View.PresentedName;
    return rejectSourceObject(HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS, Failure);
  }
  hsa_isa_t SourceIsaHandle{};
  bool Compatible = false;
  if (State->NextIsaFromName(SourceIsa.c_str(), &SourceIsaHandle) !=
          HSA_STATUS_SUCCESS ||
      !SourceIsaHandle.handle ||
      State->NextIsaCompatible(SourceIsaHandle, View.PresentedIsa,
                               &Compatible) != HSA_STATUS_SUCCESS ||
      !Compatible) {
    Failure = "source ISA " + SourceIsa +
              " is not compatible with presented ISA " + View.PresentedName;
    return rejectSourceObject(HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS, Failure);
  }

  std::shared_ptr<SourceObject> Object(new (std::nothrow) SourceObject);
  if (!Object)
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  Object->SourceElf = std::move(Source);
  Object->Agent = Agent;
  Object->Parent = Executable;
  Object->Profile = Profile;
  Object->Rounding = Rounding;
  Object->Options = Options ? Options : "";
  Object->SourceIsa = std::move(SourceIsa);
  Object->TargetIsa = View.ExecutionName;
  Object->SourceGfx = View.PresentedGfx;
  Object->TargetGfx = View.ExecutionGfx;
  Object->TargetMaxWorkgroupDim = View.MaxWorkgroupDim;
  Object->TargetMaxWorkgroupSize = View.MaxWorkgroupSize;
  Object->TargetMaxGridDim = View.MaxGridDim;
  Object->TargetMaxGridSize = View.MaxGridSize;

  llvm::StringSet<> Names;
  Object->Kernels.reserve(Metadata.size());
  for (KernelMetadata &Entry : Metadata) {
    if (!Names.insert(Entry.Name).second ||
        !Names.insert(Entry.Symbol).second) {
      Failure = "duplicate source kernel symbol '" + Entry.Symbol + "'";
      return rejectSourceObject(HSA_STATUS_ERROR_INVALID_CODE_OBJECT, Failure);
    }
    std::shared_ptr<KernelRecord> Kernel(new (std::nothrow) KernelRecord);
    if (!Kernel)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    if (Entry.DynamicCallstack) {
      Failure = "source kernel '" + Entry.Name +
                "' uses a dynamic call stack, which per-kernel translation "
                "does not support";
      return rejectSourceObject(HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS,
                                Failure);
    }
    std::unique_ptr<VirtualSymbolToken> SymbolToken(new (std::nothrow)
                                                        VirtualSymbolToken);
    std::unique_ptr<VirtualKernelToken> KernelToken(new (std::nothrow)
                                                        VirtualKernelToken);
    if (!SymbolToken || !KernelToken)
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    if (Entry.WavefrontSize != View.WavefrontSize) {
      Failure = "source kernel '" + Entry.Name + "' uses wave" +
                std::to_string(Entry.WavefrontSize) +
                " but the presented ISA exposes wave" +
                std::to_string(View.WavefrontSize);
      return rejectSourceObject(HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS,
                                Failure);
    }
    Kernel->Object = Object;
    Kernel->MetadataName = std::move(Entry.Name);
    Kernel->SymbolName = std::move(Entry.Symbol);
    Kernel->KernargSegmentSize = Entry.KernargSegmentSize;
    Kernel->KernargSegmentAlignment = Entry.KernargSegmentAlignment;
    Kernel->SourcePrivateSegmentSize = Entry.PrivateSegmentSize;
    Kernel->SourceGroupSegmentSize = Entry.GroupSegmentSize;
    Kernel->SourceWavefrontSize = Entry.WavefrontSize;
    Kernel->SourceDynamicCallstack = Entry.DynamicCallstack;
    Kernel->SymbolToken = std::move(SymbolToken);
    Kernel->KernelToken = std::move(KernelToken);
    Object->Kernels.push_back(std::move(Kernel));
  }

  if (!commitSourceRegistration(Executable, Object))
    refuse("source registration reservation could not be committed");
  Registration.release();
  State->Count.RegisteredObjects.fetch_add(1, std::memory_order_relaxed);
  State->Count.RegisteredKernels.fetch_add(Object->Kernels.size(),
                                           std::memory_order_relaxed);
  proofOrRefuse("\"event\":\"source_object_registered\",\"source_isa\":\"" +
                jsonEscape(Object->SourceIsa) + "\",\"target_isa\":\"" +
                jsonEscape(Object->TargetIsa) + "\",\"kernel_count\":" +
                std::to_string(Object->Kernels.size()));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolLoadProgram(hsa_executable_t Executable,
                                     hsa_code_object_reader_t Reader,
                                     const char *Options,
                                     hsa_loaded_code_object_t *Loaded) {
  if (!beginNativeMutation(Executable))
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  const hsa_status_t Status =
      State->NextLoadProgram(Executable, Reader, Options, Loaded);
  finishNativeMutation(Executable, Status);
  return Status;
}

hsa_status_t HSA_API toolExecutableFreeze(hsa_executable_t Executable,
                                          const char *Options) {
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    NativeExecutableRecord &Record =
        State->NativeExecutables[Executable.handle];
    if (Record.ActiveMutations != 0 || Record.SourceRegistrationActive ||
        Record.FreezeActive || Record.Destroying)
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    Record.ActiveMutations = 1;
    Record.FreezeActive = true;
  }
  const hsa_status_t Status = State->NextExecutableFreeze(Executable, Options);
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto Record = State->NativeExecutables.find(Executable.handle);
    if (Record == State->NativeExecutables.end() ||
        Record->second.ActiveMutations != 1 || !Record->second.FreezeActive ||
        Record->second.Destroying)
      refuse("executable freeze bookkeeping was lost");
    Record->second.ActiveMutations = 0;
    Record->second.FreezeActive = false;
    auto It = State->Executables.find(Executable.handle);
    if (Status == HSA_STATUS_SUCCESS && It != State->Executables.end())
      It->second.Frozen = true;
    if (!Record->second.HasContent && !Record->second.SourceRegistrationActive)
      State->NativeExecutables.erase(Record);
  }
  return Status;
}

hsa_status_t validateSourceExecutable(hsa_executable_t Executable,
                                      uint32_t *Result,
                                      hsa_status_t NativeStatus) {
  if (NativeStatus != HSA_STATUS_SUCCESS || !Result || *Result != 0)
    return NativeStatus;

  std::shared_ptr<SourceObject> Object;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto It = State->Executables.find(Executable.handle);
    if (It == State->Executables.end())
      return NativeStatus;
    if (!It->second.Frozen)
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    Object = It->second.Object;
  }
  if (!Object->Alive.load(std::memory_order_acquire)) {
    *Result = 1;
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolExecutableValidate(hsa_executable_t Executable,
                                            uint32_t *Result) {
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  return validateSourceExecutable(
      Executable, Result, State->NextExecutableValidate(Executable, Result));
}

hsa_status_t HSA_API toolExecutableValidateAlt(hsa_executable_t Executable,
                                               const char *Options,
                                               uint32_t *Result) {
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  return validateSourceExecutable(
      Executable, Result,
      State->NextExecutableValidateAlt(Executable, Options, Result));
}

hsa_status_t HSA_API toolGetSymbolByName(hsa_executable_t Executable,
                                         const char *Name,
                                         const hsa_agent_t *Agent,
                                         hsa_executable_symbol_t *Symbol) {
  if (!Name || !Symbol)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  bool Frozen = false;
  std::shared_ptr<KernelRecord> Kernel =
      findSourceKernel(Executable, Name, Agent, &Frozen);
  if (!Kernel)
    return State->NextGetSymbolByName(Executable, Name, Agent, Symbol);
  if (!Frozen)
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  Symbol->handle = reinterpret_cast<uint64_t>(Kernel->SymbolToken.get());
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolGetSymbol(hsa_executable_t Executable,
                                   const char *ModuleName,
                                   const char *SymbolName, hsa_agent_t Agent,
                                   int32_t CallConvention,
                                   hsa_executable_symbol_t *Symbol) {
  if (!SymbolName || !Symbol)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  bool Frozen = false;
  std::shared_ptr<KernelRecord> Kernel = findSourceKernel(
      Executable, SymbolName, Agent.handle ? &Agent : nullptr, &Frozen);
  if (!Kernel)
    return State->NextGetSymbol(Executable, ModuleName, SymbolName, Agent,
                                CallConvention, Symbol);
  if (ModuleName || CallConvention != 0)
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  if (!Frozen)
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  Symbol->handle = reinterpret_cast<uint64_t>(Kernel->SymbolToken.get());
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolSymbolGetInfo(hsa_executable_symbol_t Symbol,
                                       hsa_executable_symbol_info_t Attribute,
                                       void *Value) {
  if (!Value)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  std::shared_lock<std::shared_mutex> DispatchLock(State->DispatchMutex);
  std::shared_ptr<KernelRecord> Kernel;
  bool IsVirtual = false;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto It = State->SymbolTokens.find(Symbol.handle);
    if (It != State->SymbolTokens.end()) {
      IsVirtual = true;
      Kernel = It->second;
    }
  }
  if (!IsVirtual)
    return State->NextSymbolGetInfo(Symbol, Attribute, Value);
  if (!Kernel)
    return HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL;

  switch (static_cast<uint32_t>(Attribute)) {
  case HSA_EXECUTABLE_SYMBOL_INFO_TYPE:
    *static_cast<hsa_symbol_kind_t *>(Value) = HSA_SYMBOL_KIND_KERNEL;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH:
    *static_cast<uint32_t *>(Value) =
        static_cast<uint32_t>(Kernel->SymbolName.size());
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_NAME:
    std::memcpy(Value, Kernel->SymbolName.data(), Kernel->SymbolName.size());
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH:
    *static_cast<uint32_t *>(Value) = 0;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME:
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_AGENT: {
    const std::shared_ptr<SourceObject> Object = Kernel->Object.lock();
    if (!Object)
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL;
    *static_cast<hsa_agent_t *>(Value) = Object->Agent;
    return HSA_STATUS_SUCCESS;
  }
  case HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE:
    *static_cast<hsa_symbol_linkage_t *>(Value) = HSA_SYMBOL_LINKAGE_PROGRAM;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION:
    *static_cast<bool *>(Value) = true;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:
    *static_cast<uint64_t *>(Value) =
        reinterpret_cast<uint64_t>(Kernel->KernelToken.get());
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
    *static_cast<uint32_t *>(Value) = Kernel->SourcePrivateSegmentSize;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
    *static_cast<uint32_t *>(Value) = Kernel->SourceGroupSegmentSize;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
    *static_cast<uint32_t *>(Value) = Kernel->KernargSegmentSize;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:
    *static_cast<uint32_t *>(Value) =
        std::max<uint32_t>(16, Kernel->KernargSegmentAlignment);
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK:
    *static_cast<bool *>(Value) = Kernel->SourceDynamicCallstack;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION:
    *static_cast<uint32_t *>(Value) = 0;
    return HSA_STATUS_SUCCESS;
  case HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE:
    *static_cast<uint32_t *>(Value) = Kernel->SourceWavefrontSize;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS:
  case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION:
  case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT:
  case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT:
  case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE:
  case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST:
  case HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT:
  case HSA_EXECUTABLE_SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  default:
    if (!ensureTranslated(Kernel))
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    return State->NextSymbolGetInfo(Kernel->Symbol, Attribute, Value);
  }
}

struct IterateData {
  hsa_executable_t Parent{};
  hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t,
                           void *) = nullptr;
  void *Data = nullptr;
};

hsa_status_t iterateSourceKernels(const std::shared_ptr<SourceObject> &Object,
                                  IterateData &Data) {
  for (const std::shared_ptr<KernelRecord> &Kernel : Object->Kernels) {
    if (!Object->Alive.load(std::memory_order_acquire))
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    const hsa_executable_symbol_t Symbol{
        reinterpret_cast<uint64_t>(Kernel->SymbolToken.get())};
    const hsa_status_t Status = Data.Callback(Data.Parent, Symbol, Data.Data);
    if (Status != HSA_STATUS_SUCCESS)
      return Status;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolIterateSymbols(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *),
    void *Data) {
  if (!Callback)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const hsa_status_t NativeStatus =
      State->NextIterateSymbols(Executable, Callback, Data);
  if (NativeStatus != HSA_STATUS_SUCCESS)
    return NativeStatus;
  std::shared_ptr<SourceObject> Object;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto It = State->Executables.find(Executable.handle);
    if (It != State->Executables.end()) {
      if (!It->second.Frozen)
        return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
      Object = It->second.Object;
    }
  }
  if (!Object)
    return HSA_STATUS_SUCCESS;
  std::unique_ptr<SourceIterationLease> Lease =
      SourceIterationLease::acquire(Object);
  if (!Lease)
    return Object->Alive.load(std::memory_order_acquire)
               ? HSA_STATUS_ERROR_OUT_OF_RESOURCES
               : HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  IterateData Iteration{Executable, Callback, Data};
  return iterateSourceKernels(Object, Iteration);
}

struct AgentIterateData {
  hsa_executable_t Parent{};
  hsa_agent_t Agent{};
  hsa_status_t (*Callback)(hsa_executable_t, hsa_agent_t,
                           hsa_executable_symbol_t, void *) = nullptr;
  void *Data = nullptr;
};

hsa_status_t HSA_API toolIterateAgentSymbols(
    hsa_executable_t Executable, hsa_agent_t Agent,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_agent_t,
                             hsa_executable_symbol_t, void *),
    void *Data) {
  if (!Callback)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const hsa_status_t NativeStatus =
      State->NextIterateAgentSymbols(Executable, Agent, Callback, Data);
  if (NativeStatus != HSA_STATUS_SUCCESS)
    return NativeStatus;
  std::shared_ptr<SourceObject> Object;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto It = State->Executables.find(Executable.handle);
    if (It != State->Executables.end() &&
        It->second.Object->Agent.handle == Agent.handle) {
      if (!It->second.Frozen)
        return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
      Object = It->second.Object;
    }
  }
  if (!Object)
    return HSA_STATUS_SUCCESS;
  std::unique_ptr<SourceIterationLease> Lease =
      SourceIterationLease::acquire(Object);
  if (!Lease)
    return Object->Alive.load(std::memory_order_acquire)
               ? HSA_STATUS_ERROR_OUT_OF_RESOURCES
               : HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  for (const std::shared_ptr<KernelRecord> &Kernel : Object->Kernels) {
    if (!Object->Alive.load(std::memory_order_acquire))
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    const hsa_executable_symbol_t Symbol{
        reinterpret_cast<uint64_t>(Kernel->SymbolToken.get())};
    const hsa_status_t Status = Callback(Executable, Agent, Symbol, Data);
    if (Status != HSA_STATUS_SUCCESS)
      return Status;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t HSA_API toolIterateProgramSymbols(
    hsa_executable_t Executable,
    hsa_status_t (*Callback)(hsa_executable_t, hsa_executable_symbol_t, void *),
    void *Data) {
  return State->NextIterateProgramSymbols(Executable, Callback, Data);
}

hsa_status_t HSA_API toolExecutableDestroy(hsa_executable_t Executable) {
  std::shared_ptr<SourceObject> Object;
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto Native = State->NativeExecutables.find(Executable.handle);
    if (Native != State->NativeExecutables.end() &&
        (Native->second.Destroying || Native->second.ActiveMutations != 0 ||
         Native->second.SourceRegistrationActive ||
         Native->second.FreezeActive))
      return HSA_STATUS_ERROR_INVALID_EXECUTABLE;
    auto It = State->Executables.find(Executable.handle);
    if (It != State->Executables.end()) {
      Object = It->second.Object;
      State->Executables.erase(It);
    }
    NativeExecutableRecord &Record =
        State->NativeExecutables[Executable.handle];
    Record.Destroying = true;
  }
  const bool Cleanup = Object && requestSourceObjectCleanup(Object);
  if (Cleanup)
    cleanupSourceObject(Object);
  const hsa_status_t Status = State->NextExecutableDestroy(Executable);
  if (Object) {
    if (Status != HSA_STATUS_SUCCESS)
      refuse("runtime failed to destroy a registered parent executable");
  }
  {
    std::lock_guard<std::mutex> Lock(State->Mutex);
    auto It = State->NativeExecutables.find(Executable.handle);
    if (It == State->NativeExecutables.end() || !It->second.Destroying ||
        It->second.ActiveMutations != 0)
      refuse("executable destruction bookkeeping was lost");
    if (Status == HSA_STATUS_SUCCESS || !It->second.HasContent)
      State->NativeExecutables.erase(It);
    else
      It->second.Destroying = false;
  }
  return Status;
}

} // namespace
} // namespace COMGR::hotswap::hsa_tool
