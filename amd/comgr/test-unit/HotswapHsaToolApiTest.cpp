//===- HotswapHsaToolApiTest.cpp -----------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-object.h"
#include "hotswap-platform-io.h"
#include "hotswap-proof.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ext_amd.h>
#include <hsa_ven_amd_loader.h>

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

extern "C" bool OnLoad(HsaApiTable *Table, uint64_t RuntimeVersion,
                       uint64_t FailedToolCount,
                       const char *const *FailedToolNames);
extern "C" void OnUnload();

namespace {

static std::string readFile(llvm::StringRef Path) {
  auto Buffer = llvm::MemoryBuffer::getFile(Path);
  if (!Buffer) {
    ADD_FAILURE() << "cannot read " << Path.str() << ": "
                  << Buffer.getError().message();
    return {};
  }
  return (*Buffer)->getBuffer().str();
}

TEST(HotswapHsaToolProof, EscapesEveryJsonControlByte) {
  std::string Input;
  for (unsigned C = 0; C != 0x20; ++C)
    Input += static_cast<char>(C);
  Input += '"';
  Input += '\\';

  EXPECT_EQ(COMGR::hotswap::hsa_tool::jsonEscape(Input),
            "\\u0000\\u0001\\u0002\\u0003\\u0004\\u0005\\u0006\\u0007"
            "\\b\\t\\n\\u000b\\f\\r\\u000e\\u000f"
            "\\u0010\\u0011\\u0012\\u0013\\u0014\\u0015\\u0016\\u0017"
            "\\u0018\\u0019\\u001a\\u001b\\u001c\\u001d\\u001e\\u001f"
            "\\\"\\\\");
}

TEST(HotswapHsaToolProof, ReplacesInvalidUtf8BeforeLogging) {
  const char Invalid[] = {static_cast<char>(0x80), static_cast<char>(0xff)};
  EXPECT_EQ(COMGR::hotswap::hsa_tool::jsonEscape(
                llvm::StringRef(Invalid, sizeof(Invalid))),
            "\xef\xbf\xbd\xef\xbf\xbd");
}

TEST(HotswapHsaToolProof, ConcurrentProcessesAppendWholeRecords) {
  char Path[] = "/tmp/comgr-hotswap-proof-processes-XXXXXX";
  const int File = mkstemp(Path);
  ASSERT_NE(File, -1);
  ASSERT_EQ(close(File), 0);

  constexpr unsigned ProcessCount = 8;
  constexpr unsigned RecordCount = 100;
  for (unsigned Process = 0; Process != ProcessCount; ++Process) {
    const pid_t Child = fork();
    ASSERT_NE(Child, -1);
    if (Child == 0) {
      for (unsigned Record = 0; Record != RecordCount; ++Record) {
        const std::string Fields =
            "\"event\":\"child\",\"child\":" + std::to_string(Process) +
            ",\"record\":" + std::to_string(Record);
        if (!COMGR::hotswap::hsa_tool::appendProofLine(
                {Path, Fields, static_cast<uint64_t>(getpid())}))
          _exit(1);
      }
      _exit(0);
    }
  }

  for (unsigned Process = 0; Process != ProcessCount; ++Process) {
    int Status = 0;
    ASSERT_NE(wait(&Status), -1);
    ASSERT_TRUE(WIFEXITED(Status));
    ASSERT_EQ(WEXITSTATUS(Status), 0);
  }

  const int ReadFile = open(Path, O_RDONLY);
  ASSERT_NE(ReadFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Contents =
      COMGR::hotswap::hsa_tool::readWholeFile(ReadFile);
  ASSERT_TRUE(Contents);
  ASSERT_EQ(close(ReadFile), 0);
  ASSERT_EQ(unlink(Path), 0);

  unsigned Lines = 0;
  size_t Begin = 0;
  while (Begin != Contents->size()) {
    const auto End =
        std::find(Contents->begin() + Begin, Contents->end(), '\n');
    ASSERT_NE(End, Contents->end());
    const std::string Line(Contents->begin() + Begin, End);
    EXPECT_EQ(Line.rfind("{\"pid\":", 0), 0u);
    EXPECT_NE(Line.find(",\"event\":\"child\",\"child\":"), std::string::npos);
    EXPECT_EQ(Line.back(), '}');
    ++Lines;
    Begin = static_cast<size_t>(End - Contents->begin()) + 1;
  }
  EXPECT_EQ(Lines, ProcessCount * RecordCount);
}

constexpr hsa_agent_t GpuAgent{1};
constexpr hsa_isa_t PhysicalIsa{942};
constexpr hsa_isa_t PresentedIsa{1250};
constexpr hsa_isa_t FixtureIsa{950};
constexpr hsa_wavefront_t PresentedWave{32};
constexpr hsa_wavefront_t FixtureWave{64};

bool FailExecutionIsa = false;
bool IsaNameLengthIncludesTerminator = false;
uint32_t PresentedWavefrontSize = 32;
unsigned NativeQueueCreateCalls = 0;
unsigned SoftQueueCreateCalls = 0;
unsigned InterceptQueueCreateCalls = 0;
unsigned InterceptRegisterCalls = 0;
unsigned QueueGetInfoCalls = 0;
unsigned PriorityCalls = 0;
unsigned CuMaskCalls = 0;
unsigned WriterCalls = 0;
unsigned ReaderMemoryCalls = 0;
unsigned ReaderFileCalls = 0;
unsigned ReaderFileSliceCalls = 0;
unsigned ReaderDestroyCalls = 0;
unsigned ExecutableCreateCalls = 0;
unsigned ExecutableDestroyCalls = 0;
unsigned TranslatedLoadCalls = 0;
uint64_t WrittenPacketCount = 0;
uint64_t NextReaderHandle = 100;
uint64_t NextExecutableHandle = 1000;
uint32_t ReturnedQueueSize = 0;
bool EnableTranslatedLoader = false;
bool TranslatedDynamicCallstack = false;
uint32_t TranslatedKernargSize = 288;
bool EnableNativeProgramLoad = false;
bool EnableGlobalDefinitions = false;
bool AllowNullLoaderTableSuccess = false;
hsa_status_t ExecutableDestroyResult = HSA_STATUS_SUCCESS;
hsa_status_t ExecutableCreateResult = HSA_STATUS_SUCCESS;
std::mutex MutationMutex;
std::condition_variable MutationCondition;
bool BlockNativeProgramLoad = false;
bool NativeProgramLoadEntered = false;
bool ReleaseNativeProgramLoad = false;
bool BlockExecutableGetInfo = false;
bool ExecutableGetInfoEntered = false;
bool ReleaseExecutableGetInfo = false;
bool BlockExecutableFreeze = false;
bool ExecutableFreezeEntered = false;
bool ReleaseExecutableFreeze = false;
hsa_kernel_dispatch_packet_t WrittenPacket{};
hsa_amd_queue_intercept_handler RegisteredInterceptor = nullptr;
void *RegisteredInterceptorData = nullptr;
hsa_queue_t Queue{};

hsa_status_t fakeIterateAgents(hsa_status_t (*Callback)(hsa_agent_t, void *),
                               void *Data) {
  return Callback(GpuAgent, Data);
}

hsa_status_t fakeAgentGetInfo(hsa_agent_t Agent, hsa_agent_info_t Attribute,
                              void *Value) {
  if (Agent.handle != GpuAgent.handle || !Value)
    return HSA_STATUS_ERROR_INVALID_AGENT;
  if (Attribute == HSA_AGENT_INFO_DEVICE) {
    *static_cast<hsa_device_type_t *>(Value) = HSA_DEVICE_TYPE_GPU;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute ==
      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_EXECUTION_ISA)) {
    if (FailExecutionIsa)
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    *static_cast<hsa_isa_t *>(Value) = PhysicalIsa;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_AGENT_INFO_PROFILE) {
    *static_cast<hsa_profile_t *>(Value) = HSA_PROFILE_FULL;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE) {
    *static_cast<hsa_default_float_rounding_mode_t *>(Value) =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_AGENT_INFO_WORKGROUP_MAX_DIM) {
    llvm::MutableArrayRef<uint16_t> Dimensions(static_cast<uint16_t *>(Value),
                                               3);
    Dimensions[0] = 1024;
    Dimensions[1] = 1024;
    Dimensions[2] = 1024;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_AGENT_INFO_WORKGROUP_MAX_SIZE) {
    *static_cast<uint32_t *>(Value) = 1024;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t outerAgentGetInfo(hsa_agent_t Agent, hsa_agent_info_t Attribute,
                               void *Value) {
  return fakeAgentGetInfo(Agent, Attribute, Value);
}

hsa_status_t fakeIsaFromName(const char *Name, hsa_isa_t *Isa) {
  if (!Name || !Isa)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (std::strstr(Name, "gfx1250"))
    *Isa = PresentedIsa;
  else if (std::strstr(Name, "gfx950"))
    *Isa = FixtureIsa;
  else if (std::strstr(Name, "gfx942"))
    *Isa = PhysicalIsa;
  else
    return HSA_STATUS_ERROR_INVALID_ISA_NAME;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeIsaCompatible(hsa_isa_t, hsa_isa_t, bool *Compatible) {
  if (!Compatible)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *Compatible = true;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeAgentIterateIsas(hsa_agent_t,
                                  hsa_status_t (*Callback)(hsa_isa_t, void *),
                                  void *Data) {
  return Callback(PhysicalIsa, Data);
}

const char *nameForIsa(hsa_isa_t Isa) {
  if (Isa.handle == PresentedIsa.handle)
    return "amdgcn-amd-amdhsa--gfx1250";
  if (Isa.handle == FixtureIsa.handle)
    return "amdgcn-amd-amdhsa--gfx950";
  if (Isa.handle == PhysicalIsa.handle)
    return "amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-";
  return nullptr;
}

hsa_status_t fakeIsaGetInfo(hsa_isa_t Isa, hsa_isa_info_t Attribute,
                            void *Value) {
  const char *Name = nameForIsa(Isa);
  if (!Name || !Value)
    return HSA_STATUS_ERROR_INVALID_ISA;
  if (Attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(Value) =
        std::strlen(Name) + (IsaNameLengthIncludesTerminator ? 1 : 0);
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_ISA_INFO_NAME) {
    const size_t NameLength = std::strlen(Name);
    llvm::MutableArrayRef<char> Destination(
        static_cast<char *>(Value),
        NameLength + (IsaNameLengthIncludesTerminator ? 1 : 0));
    std::memcpy(Destination.data(), Name, NameLength);
    if (IsaNameLengthIncludesTerminator)
      Destination.back() = '\0';
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES) {
    llvm::MutableArrayRef<bool> Modes(static_cast<bool *>(Value), 4);
    Modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT] = true;
    Modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_ZERO] = true;
    Modes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR] = true;
    return HSA_STATUS_SUCCESS;
  }
  if (Isa.handle == PhysicalIsa.handle &&
      Attribute == HSA_ISA_INFO_WORKGROUP_MAX_DIM) {
    llvm::MutableArrayRef<uint16_t> Dimensions(static_cast<uint16_t *>(Value),
                                               3);
    Dimensions[0] = 1024;
    Dimensions[1] = 1024;
    Dimensions[2] = 1024;
    return HSA_STATUS_SUCCESS;
  }
  if (Isa.handle == PhysicalIsa.handle &&
      Attribute == HSA_ISA_INFO_WORKGROUP_MAX_SIZE) {
    *static_cast<uint32_t *>(Value) = 1024;
    return HSA_STATUS_SUCCESS;
  }
  if (Isa.handle == PhysicalIsa.handle &&
      Attribute == HSA_ISA_INFO_GRID_MAX_DIM) {
    *static_cast<hsa_dim3_t *>(Value) = {std::numeric_limits<uint32_t>::max(),
                                         std::numeric_limits<uint32_t>::max(),
                                         std::numeric_limits<uint32_t>::max()};
    return HSA_STATUS_SUCCESS;
  }
  if (Isa.handle == PhysicalIsa.handle &&
      Attribute == HSA_ISA_INFO_GRID_MAX_SIZE) {
    *static_cast<uint64_t *>(Value) = std::numeric_limits<uint64_t>::max();
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t fakeIsaIterateWavefronts(hsa_isa_t Isa,
                                      hsa_status_t (*Callback)(hsa_wavefront_t,
                                                               void *),
                                      void *Data) {
  if (Isa.handle == PresentedIsa.handle)
    return Callback(PresentedWave, Data);
  if (Isa.handle == FixtureIsa.handle)
    return Callback(FixtureWave, Data);
  else
    return HSA_STATUS_ERROR_INVALID_ISA;
}

hsa_status_t fakeWavefrontGetInfo(hsa_wavefront_t Wave,
                                  hsa_wavefront_info_t Attribute, void *Value) {
  if (Attribute != HSA_WAVEFRONT_INFO_SIZE || !Value)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (Wave.handle == PresentedWave.handle)
    *static_cast<uint32_t *>(Value) = PresentedWavefrontSize;
  else if (Wave.handle == FixtureWave.handle)
    *static_cast<uint32_t *>(Value) = 64;
  else
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeIsaGetExceptionPolicies(hsa_isa_t Isa, hsa_profile_t,
                                         uint16_t *Mask) {
  if (Isa.handle != PresentedIsa.handle || !Mask)
    return HSA_STATUS_ERROR_INVALID_ISA;
  *Mask = HSA_EXCEPTION_POLICY_BREAK | HSA_EXCEPTION_POLICY_DETECT;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeIsaGetRoundMethod(hsa_isa_t Isa, hsa_fp_type_t,
                                   hsa_flush_mode_t,
                                   hsa_round_method_t *Method) {
  if (Isa.handle != PresentedIsa.handle || !Method)
    return HSA_STATUS_ERROR_INVALID_ISA;
  *Method = HSA_ROUND_METHOD_SINGLE;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t
fakeNativeQueueCreate(hsa_agent_t, uint32_t Size, hsa_queue_type32_t,
                      void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
                      uint32_t, uint32_t, hsa_queue_t **Result) {
  ++NativeQueueCreateCalls;
  Queue.size = ReturnedQueueSize ? ReturnedQueueSize : Size;
  *Result = &Queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t
fakeInterceptQueueCreate(hsa_agent_t, uint32_t Size, hsa_queue_type32_t,
                         void (*)(hsa_status_t, hsa_queue_t *, void *), void *,
                         uint32_t, uint32_t, hsa_queue_t **Result) {
  ++InterceptQueueCreateCalls;
  Queue.size = ReturnedQueueSize ? ReturnedQueueSize : Size;
  *Result = &Queue;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeSoftQueueCreate(hsa_region_t, uint32_t, hsa_queue_type32_t,
                                 uint32_t, hsa_signal_t, hsa_queue_t **) {
  ++SoftQueueCreateCalls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeInterceptRegister(hsa_queue_t *ProtectedQueue,
                                   hsa_amd_queue_intercept_handler Callback,
                                   void *Data) {
  if (ProtectedQueue != &Queue || !Callback)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  ++InterceptRegisterCalls;
  RegisteredInterceptor = Callback;
  RegisteredInterceptorData = Data;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeQueueDestroy(hsa_queue_t *DestroyedQueue) {
  if (DestroyedQueue != &Queue)
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  RegisteredInterceptor = nullptr;
  RegisteredInterceptorData = nullptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeQueueGetInfo(hsa_queue_t *, hsa_queue_info_attribute_t,
                              void *) {
  ++QueueGetInfoCalls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeQueueSetPriority(hsa_queue_t *PriorityQueue,
                                  hsa_amd_queue_priority_t) {
  if (PriorityQueue != &Queue)
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  ++PriorityCalls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeQueueSetCuMask(const hsa_queue_t *MaskQueue, uint32_t,
                                const uint32_t *) {
  if (MaskQueue != &Queue)
    return HSA_STATUS_ERROR_INVALID_QUEUE;
  ++CuMaskCalls;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeAmdQueueCreate(hsa_agent_t, hsa_amd_queue_create_desc_t *,
                                uint32_t) {
  return HSA_STATUS_ERROR;
}

void fakePacketWriter(const void *Packets, uint64_t Count) {
  ++WriterCalls;
  WrittenPacketCount = Count;
  if (Count)
    WrittenPacket = *static_cast<const hsa_kernel_dispatch_packet_t *>(Packets);
}

hsa_status_t fakeReaderMemory(const void *, size_t,
                              hsa_code_object_reader_t *Reader) {
  ++ReaderMemoryCalls;
  Reader->handle = NextReaderHandle++;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeReaderFile(hsa_file_t, hsa_code_object_reader_t *Reader) {
  ++ReaderFileCalls;
  Reader->handle = NextReaderHandle++;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeReaderDestroy(hsa_code_object_reader_t) {
  ++ReaderDestroyCalls;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeExecutableCreate(hsa_profile_t,
                                  hsa_default_float_rounding_mode_t,
                                  const char *, hsa_executable_t *Executable) {
  if (!EnableTranslatedLoader)
    return HSA_STATUS_ERROR;
  ++ExecutableCreateCalls;
  if (ExecutableCreateResult != HSA_STATUS_SUCCESS)
    return ExecutableCreateResult;
  Executable->handle = NextExecutableHandle++;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeExecutableDestroy(hsa_executable_t) {
  ++ExecutableDestroyCalls;
  return ExecutableDestroyResult;
}
hsa_status_t fakeExecutableFreeze(hsa_executable_t, const char *) {
  std::unique_lock<std::mutex> Lock(MutationMutex);
  if (BlockExecutableFreeze) {
    ExecutableFreezeEntered = true;
    MutationCondition.notify_all();
    MutationCondition.wait(Lock, [] { return ReleaseExecutableFreeze; });
  }
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeExecutableValidate(hsa_executable_t, uint32_t *Result) {
  if (!Result)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *Result = 0;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeExecutableValidateAlt(hsa_executable_t, const char *,
                                       uint32_t *Result) {
  return fakeExecutableValidate(hsa_executable_t{}, Result);
}
hsa_status_t fakeExecutableGetInfo(hsa_executable_t, hsa_executable_info_t,
                                   void *);
hsa_status_t fakeExecutableGetInfo(hsa_executable_t,
                                   hsa_executable_info_t Attribute,
                                   void *Value) {
  {
    std::unique_lock<std::mutex> Lock(MutationMutex);
    if (BlockExecutableGetInfo) {
      ExecutableGetInfoEntered = true;
      MutationCondition.notify_all();
      MutationCondition.wait(Lock, [] { return ReleaseExecutableGetInfo; });
    }
  }
  if (!Value)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (Attribute == HSA_EXECUTABLE_INFO_STATE) {
    *static_cast<hsa_executable_state_t *>(Value) =
        HSA_EXECUTABLE_STATE_UNFROZEN;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_EXECUTABLE_INFO_PROFILE) {
    *static_cast<hsa_profile_t *>(Value) = HSA_PROFILE_FULL;
    return HSA_STATUS_SUCCESS;
  }
  if (Attribute == HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE) {
    *static_cast<hsa_default_float_rounding_mode_t *>(Value) =
        HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}
hsa_status_t fakeLoadProgram(hsa_executable_t, hsa_code_object_reader_t,
                             const char *, hsa_loaded_code_object_t *) {
  {
    std::unique_lock<std::mutex> Lock(MutationMutex);
    if (BlockNativeProgramLoad) {
      NativeProgramLoadEntered = true;
      MutationCondition.notify_all();
      MutationCondition.wait(Lock, [] { return ReleaseNativeProgramLoad; });
    }
  }
  return EnableNativeProgramLoad ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}
hsa_status_t fakeLoadAgent(hsa_executable_t, hsa_agent_t,
                           hsa_code_object_reader_t, const char *,
                           hsa_loaded_code_object_t *Loaded) {
  if (!EnableTranslatedLoader)
    return HSA_STATUS_ERROR;
  ++TranslatedLoadCalls;
  if (Loaded)
    Loaded->handle = 0x700 + TranslatedLoadCalls;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeLoadCodeObject(hsa_executable_t, hsa_agent_t,
                                hsa_code_object_t, const char *) {
  return HSA_STATUS_ERROR;
}
hsa_status_t fakeDefineGlobal(hsa_executable_t, const char *, void *) {
  return EnableGlobalDefinitions ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}
hsa_status_t fakeDefineAgentGlobal(hsa_executable_t, hsa_agent_t, const char *,
                                   void *) {
  return EnableGlobalDefinitions ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}
hsa_status_t fakeDefineReadonly(hsa_executable_t, hsa_agent_t, const char *,
                                void *) {
  return EnableGlobalDefinitions ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}
hsa_status_t fakeGetSymbolByName(hsa_executable_t Executable, const char *,
                                 const hsa_agent_t *,
                                 hsa_executable_symbol_t *Symbol) {
  if (!EnableTranslatedLoader)
    return HSA_STATUS_ERROR;
  Symbol->handle = 0x500 + Executable.handle;
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeGetSymbol(hsa_executable_t, const char *, const char *,
                           hsa_agent_t, int32_t, hsa_executable_symbol_t *) {
  return HSA_STATUS_ERROR;
}
hsa_status_t fakeSymbolGetInfo(hsa_executable_symbol_t Symbol,
                               hsa_executable_symbol_info_t Attribute,
                               void *Value) {
  if (!EnableTranslatedLoader || Symbol.handle < 0x500 || !Value)
    return HSA_STATUS_ERROR;
  switch (Attribute) {
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:
    *static_cast<uint64_t *>(Value) = 0x1234000;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE:
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE:
    *static_cast<uint32_t *>(Value) = 0;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE:
    *static_cast<uint32_t *>(Value) = TranslatedKernargSize;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT:
    // A target requiring weaker alignment than the presented ABI is safe.
    *static_cast<uint32_t *>(Value) = 8;
    return HSA_STATUS_SUCCESS;
  case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK:
    *static_cast<bool *>(Value) = TranslatedDynamicCallstack;
    return HSA_STATUS_SUCCESS;
  default:
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}
hsa_status_t fakeIterateSymbols(hsa_executable_t,
                                hsa_status_t (*)(hsa_executable_t,
                                                 hsa_executable_symbol_t,
                                                 void *),
                                void *) {
  return HSA_STATUS_SUCCESS;
}
hsa_status_t
fakeIterateAgentSymbols(hsa_executable_t, hsa_agent_t,
                        hsa_status_t (*)(hsa_executable_t, hsa_agent_t,
                                         hsa_executable_symbol_t, void *),
                        void *) {
  return HSA_STATUS_SUCCESS;
}
hsa_status_t fakeIterateProgramSymbols(hsa_executable_t,
                                       hsa_status_t (*)(hsa_executable_t,
                                                        hsa_executable_symbol_t,
                                                        void *),
                                       void *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeLoaderHostAddress(const void *DeviceAddress,
                                   const void **HostAddress) {
  if (EnableTranslatedLoader &&
      reinterpret_cast<uintptr_t>(DeviceAddress) == 0x1234000 && HostAddress) {
    *HostAddress = reinterpret_cast<const void *>(0x5678000);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t fakeLoaderQuerySegments(hsa_ven_amd_loader_segment_descriptor_t *,
                                     size_t *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeLoaderQueryExecutable(const void *, hsa_executable_t *) {
  return HSA_STATUS_ERROR_INVALID_ARGUMENT;
}

hsa_status_t fakeLoaderIterateLoaded(hsa_executable_t,
                                     hsa_status_t (*)(hsa_executable_t,
                                                      hsa_loaded_code_object_t,
                                                      void *),
                                     void *) {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeLoaderReaderFileSlice(hsa_file_t, size_t, size_t,
                                       hsa_code_object_reader_t *Reader) {
  ++ReaderFileSliceCalls;
  Reader->handle = NextReaderHandle++;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeLoaderIterateExecutables(hsa_status_t (*)(hsa_executable_t,
                                                           void *),
                                          void *) {
  return HSA_STATUS_SUCCESS;
}

template <typename TableT> void initializeLoaderBase(TableT &Table) {
  Table = {};
  Table.hsa_ven_amd_loader_query_host_address = fakeLoaderHostAddress;
  Table.hsa_ven_amd_loader_query_segment_descriptors = fakeLoaderQuerySegments;
  Table.hsa_ven_amd_loader_query_executable = fakeLoaderQueryExecutable;
}

void initializeLoaderTable(hsa_ven_amd_loader_1_00_pfn_t &Table) {
  initializeLoaderBase(Table);
}

void initializeLoaderTable(hsa_ven_amd_loader_1_01_pfn_t &Table) {
  initializeLoaderBase(Table);
  Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects =
      fakeLoaderIterateLoaded;
}

void initializeLoaderTable(hsa_ven_amd_loader_1_02_pfn_t &Table) {
  initializeLoaderBase(Table);
  Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects =
      fakeLoaderIterateLoaded;
  Table
      .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size =
      fakeLoaderReaderFileSlice;
}

void initializeLoaderTable(hsa_ven_amd_loader_1_03_pfn_t &Table) {
  initializeLoaderBase(Table);
  Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects =
      fakeLoaderIterateLoaded;
  Table
      .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size =
      fakeLoaderReaderFileSlice;
  Table.hsa_ven_amd_loader_iterate_executables = fakeLoaderIterateExecutables;
}

// The adjacent major and minor parameters are fixed by the HSA ABI.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
hsa_status_t fakeGetExtensionTable(uint16_t Extension, uint16_t Major,
                                   uint16_t Minor, void *RawTable) {
  if (!RawTable && AllowNullLoaderTableSuccess)
    return HSA_STATUS_SUCCESS;
  if (Extension != HSA_EXTENSION_AMD_LOADER || Major != 1 || !RawTable)
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  switch (Minor) {
  case 0:
    initializeLoaderTable(
        *static_cast<hsa_ven_amd_loader_1_00_pfn_t *>(RawTable));
    break;
  case 1:
    initializeLoaderTable(
        *static_cast<hsa_ven_amd_loader_1_01_pfn_t *>(RawTable));
    break;
  case 2:
    initializeLoaderTable(
        *static_cast<hsa_ven_amd_loader_1_02_pfn_t *>(RawTable));
    break;
  case 3:
    initializeLoaderTable(
        *static_cast<hsa_ven_amd_loader_1_03_pfn_t *>(RawTable));
    break;
  default:
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t fakeGetMajorExtensionTable(uint16_t Extension, uint16_t Major,
                                        size_t TableSize, void *RawTable) {
  if (!RawTable && AllowNullLoaderTableSuccess)
    return HSA_STATUS_SUCCESS;
  if (Extension != HSA_EXTENSION_AMD_LOADER || Major != 1 || !RawTable ||
      TableSize > sizeof(hsa_ven_amd_loader_1_03_pfn_t))
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_NOT_SUPPORTED);
  hsa_ven_amd_loader_1_03_pfn_t Table{};
  initializeLoaderTable(Table);
  std::memcpy(RawTable, &Table, TableSize);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t captureIsa(hsa_isa_t Isa, void *Data) {
  *static_cast<hsa_isa_t *>(Data) = Isa;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t countAgentSymbol(hsa_executable_t, hsa_agent_t,
                              hsa_executable_symbol_t, void *Data) {
  ++*static_cast<unsigned *>(Data);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t countSymbol(hsa_executable_t, hsa_executable_symbol_t,
                         void *Data) {
  ++*static_cast<unsigned *>(Data);
  return HSA_STATUS_SUCCESS;
}

struct IterationDestroyContext {
  std::mutex Mutex;
  std::condition_variable Condition;
  CoreApiTable *Core = nullptr;
  bool CallbackEntered = false;
  bool ParentDestroyed = false;
  hsa_status_t SymbolInfoStatus = HSA_STATUS_ERROR;
};

hsa_status_t querySymbolAfterParentDestroy(hsa_executable_t,
                                           hsa_executable_symbol_t Symbol,
                                           void *Data) {
  auto &Context = *static_cast<IterationDestroyContext *>(Data);
  {
    std::unique_lock<std::mutex> Lock(Context.Mutex);
    Context.CallbackEntered = true;
    Context.Condition.notify_all();
    Context.Condition.wait(Lock, [&] { return Context.ParentDestroyed; });
  }
  uint64_t KernelObject = 0;
  Context.SymbolInfoStatus = Context.Core->hsa_executable_symbol_get_info_fn(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject);
  return Context.SymbolInfoStatus;
}

class HotswapHsaToolApiTest : public testing::Test {
protected:
  CoreApiTable Core{};
  AmdExtTable Amd{};
  HsaApiTable Root{};

  void SetUp() override {
    unsetenv("HSA_HOTSWAP_PRESENT_ISA");
    unsetenv("HSA_HOTSWAP_CACHE_DIR");
    unsetenv("HSA_HOTSWAP_PROOF_LOG");
    unsetenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO");
    FailExecutionIsa = false;
    IsaNameLengthIncludesTerminator = false;
    PresentedWavefrontSize = 32;
    NativeQueueCreateCalls = 0;
    SoftQueueCreateCalls = 0;
    InterceptQueueCreateCalls = 0;
    InterceptRegisterCalls = 0;
    QueueGetInfoCalls = 0;
    PriorityCalls = 0;
    CuMaskCalls = 0;
    WriterCalls = 0;
    ReaderMemoryCalls = 0;
    ReaderFileCalls = 0;
    ReaderFileSliceCalls = 0;
    ReaderDestroyCalls = 0;
    ExecutableCreateCalls = 0;
    ExecutableDestroyCalls = 0;
    TranslatedLoadCalls = 0;
    WrittenPacketCount = 0;
    NextReaderHandle = 100;
    NextExecutableHandle = 1000;
    ReturnedQueueSize = 0;
    EnableTranslatedLoader = false;
    TranslatedDynamicCallstack = false;
    TranslatedKernargSize = 288;
    EnableNativeProgramLoad = false;
    EnableGlobalDefinitions = false;
    AllowNullLoaderTableSuccess = false;
    ExecutableDestroyResult = HSA_STATUS_SUCCESS;
    ExecutableCreateResult = HSA_STATUS_SUCCESS;
    {
      std::lock_guard<std::mutex> Lock(MutationMutex);
      BlockNativeProgramLoad = false;
      NativeProgramLoadEntered = false;
      ReleaseNativeProgramLoad = false;
      BlockExecutableGetInfo = false;
      ExecutableGetInfoEntered = false;
      ReleaseExecutableGetInfo = false;
      BlockExecutableFreeze = false;
      ExecutableFreezeEntered = false;
      ReleaseExecutableFreeze = false;
    }
    WrittenPacket = {};
    RegisteredInterceptor = nullptr;
    RegisteredInterceptorData = nullptr;

    Core.version.minor_id = sizeof(Core);
    Amd.version.minor_id = sizeof(Amd);
    Root.core_ = &Core;
    Root.amd_ext_ = &Amd;

    Core.hsa_iterate_agents_fn = fakeIterateAgents;
    Core.hsa_agent_get_info_fn = fakeAgentGetInfo;
    Core.hsa_isa_from_name_fn = fakeIsaFromName;
    Core.hsa_isa_compatible_fn = fakeIsaCompatible;
    Core.hsa_agent_iterate_isas_fn = fakeAgentIterateIsas;
    Core.hsa_isa_get_info_alt_fn = fakeIsaGetInfo;
    Core.hsa_isa_iterate_wavefronts_fn = fakeIsaIterateWavefronts;
    Core.hsa_wavefront_get_info_fn = fakeWavefrontGetInfo;
    Core.hsa_isa_get_exception_policies_fn = fakeIsaGetExceptionPolicies;
    Core.hsa_isa_get_round_method_fn = fakeIsaGetRoundMethod;
    Core.hsa_system_get_extension_table_fn = fakeGetExtensionTable;
    Core.hsa_system_get_major_extension_table_fn = fakeGetMajorExtensionTable;
    Core.hsa_queue_create_fn = fakeNativeQueueCreate;
    Core.hsa_soft_queue_create_fn = fakeSoftQueueCreate;
    Core.hsa_queue_destroy_fn = fakeQueueDestroy;

    Core.hsa_code_object_reader_create_from_memory_fn = fakeReaderMemory;
    Core.hsa_code_object_reader_create_from_file_fn = fakeReaderFile;
    Core.hsa_code_object_reader_destroy_fn = fakeReaderDestroy;
    Core.hsa_executable_create_alt_fn = fakeExecutableCreate;
    Core.hsa_executable_destroy_fn = fakeExecutableDestroy;
    Core.hsa_executable_freeze_fn = fakeExecutableFreeze;
    Core.hsa_executable_get_info_fn = fakeExecutableGetInfo;
    Core.hsa_executable_validate_fn = fakeExecutableValidate;
    Core.hsa_executable_validate_alt_fn = fakeExecutableValidateAlt;
    Core.hsa_executable_load_program_code_object_fn = fakeLoadProgram;
    Core.hsa_executable_load_agent_code_object_fn = fakeLoadAgent;
    Core.hsa_executable_load_code_object_fn = fakeLoadCodeObject;
    Core.hsa_executable_global_variable_define_fn = fakeDefineGlobal;
    Core.hsa_executable_agent_global_variable_define_fn = fakeDefineAgentGlobal;
    Core.hsa_executable_readonly_variable_define_fn = fakeDefineReadonly;
    Core.hsa_executable_get_symbol_by_name_fn = fakeGetSymbolByName;
    Core.hsa_executable_get_symbol_fn = fakeGetSymbol;
    Core.hsa_executable_symbol_get_info_fn = fakeSymbolGetInfo;
    Core.hsa_executable_iterate_symbols_fn = fakeIterateSymbols;
    Core.hsa_executable_iterate_agent_symbols_fn = fakeIterateAgentSymbols;
    Core.hsa_executable_iterate_program_symbols_fn = fakeIterateProgramSymbols;

    Amd.hsa_amd_queue_intercept_create_fn = fakeInterceptQueueCreate;
    Amd.hsa_amd_queue_intercept_register_fn = fakeInterceptRegister;
    Amd.hsa_amd_queue_get_info_fn = fakeQueueGetInfo;
    Amd.hsa_amd_queue_create_fn = fakeAmdQueueCreate;
    Amd.hsa_amd_queue_set_priority_fn = fakeQueueSetPriority;
    Amd.hsa_amd_queue_cu_set_mask_fn = fakeQueueSetCuMask;
  }

  void TearDown() override {
    if (RegisteredInterceptor)
      EXPECT_EQ(Core.hsa_queue_destroy_fn(&Queue), HSA_STATUS_SUCCESS);
    OnUnload();
    unsetenv("HSA_HOTSWAP_PRESENT_ISA");
  }

  void activate(const char *Presented = "gfx1250") {
    ASSERT_EQ(setenv("HSA_HOTSWAP_PRESENT_ISA", Presented, 1), 0);
    ASSERT_TRUE(OnLoad(&Root, HSA_API_TABLE_MAJOR_VERSION, 0, nullptr));
  }

  void createProtectedQueue() {
    hsa_queue_t *Created = nullptr;
    ASSERT_EQ(Core.hsa_queue_create_fn(GpuAgent, 64, HSA_QUEUE_TYPE_MULTI,
                                       nullptr, nullptr, 0, UINT32_MAX,
                                       &Created),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(Created, &Queue);
    ASSERT_NE(RegisteredInterceptor, nullptr);
  }
};

TEST_F(HotswapHsaToolApiTest, IsInertWithoutPresentationConfiguration) {
  auto *AgentGetInfo = Core.hsa_agent_get_info_fn;
  auto *QueueCreate = Core.hsa_queue_create_fn;
  EXPECT_TRUE(OnLoad(nullptr, HSA_API_TABLE_MAJOR_VERSION, 0, nullptr));
  EXPECT_EQ(Core.hsa_agent_get_info_fn, AgentGetInfo);
  EXPECT_EQ(Core.hsa_queue_create_fn, QueueCreate);
}

TEST_F(HotswapHsaToolApiTest, PresentsIsaAndPreservesExecutionIsa) {
  auto *IsaFromName = Core.hsa_isa_from_name_fn;
  auto *IsaGetInfo = Core.hsa_isa_get_info_alt_fn;
  auto *WavefrontGetInfo = Core.hsa_wavefront_get_info_fn;
  activate();

  hsa_isa_t Isa{};
  EXPECT_EQ(Core.hsa_agent_get_info_fn(GpuAgent, HSA_AGENT_INFO_ISA, &Isa),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Isa.handle, PresentedIsa.handle);
  char Name[64]{};
  EXPECT_EQ(Core.hsa_agent_get_info_fn(GpuAgent, HSA_AGENT_INFO_NAME, Name),
            HSA_STATUS_SUCCESS);
  EXPECT_STREQ(Name, "gfx1250");
  uint32_t WaveSize = 0;
  EXPECT_EQ(Core.hsa_agent_get_info_fn(GpuAgent, HSA_AGENT_INFO_WAVEFRONT_SIZE,
                                       &WaveSize),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(WaveSize, 32u);

  hsa_isa_t Execution{};
  EXPECT_EQ(Core.hsa_agent_get_info_fn(
                GpuAgent,
                static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_EXECUTION_ISA),
                &Execution),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Execution.handle, PhysicalIsa.handle);
  EXPECT_EQ(Core.hsa_isa_from_name_fn, IsaFromName);
  EXPECT_EQ(Core.hsa_isa_get_info_alt_fn, IsaGetInfo);
  EXPECT_EQ(Core.hsa_wavefront_get_info_fn, WavefrontGetInfo);

  hsa_isa_t Iterated{};
  EXPECT_EQ(Core.hsa_agent_iterate_isas_fn(GpuAgent, captureIsa, &Iterated),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Iterated.handle, PresentedIsa.handle);
  EXPECT_EQ(Core.hsa_isa_from_name_fn("amdgcn-amd-amdhsa--gfx1250", &Isa),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Isa.handle, PresentedIsa.handle);
  bool Compatible = false;
  EXPECT_EQ(Core.hsa_isa_compatible_fn(Isa, PresentedIsa, &Compatible),
            HSA_STATUS_SUCCESS);
  EXPECT_TRUE(Compatible);
  uint32_t NameLength = 0;
  EXPECT_EQ(
      Core.hsa_isa_get_info_alt_fn(Isa, HSA_ISA_INFO_NAME_LENGTH, &NameLength),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(NameLength, std::strlen("amdgcn-amd-amdhsa--gfx1250"));
  bool RoundingModes[3]{};
  EXPECT_EQ(Core.hsa_isa_get_info_alt_fn(
                Isa, HSA_ISA_INFO_DEFAULT_FLOAT_ROUNDING_MODES, RoundingModes),
            HSA_STATUS_SUCCESS);
  EXPECT_TRUE(RoundingModes[HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR]);
  uint16_t Policies = 0;
  EXPECT_EQ(
      Core.hsa_isa_get_exception_policies_fn(Isa, HSA_PROFILE_FULL, &Policies),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(Policies, HSA_EXCEPTION_POLICY_BREAK | HSA_EXCEPTION_POLICY_DETECT);
  hsa_round_method_t RoundMethod{};
  EXPECT_EQ(Core.hsa_isa_get_round_method_fn(
                Isa, HSA_FP_TYPE_32, HSA_FLUSH_MODE_NON_FTZ, &RoundMethod),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(RoundMethod, HSA_ROUND_METHOD_SINGLE);
  hsa_default_float_rounding_mode_t AgentRound{};
  EXPECT_EQ(
      Core.hsa_agent_get_info_fn(
          GpuAgent, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE, &AgentRound),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(AgentRound, HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR);
}

TEST_F(HotswapHsaToolApiTest, AcceptsRocrIsaNameLengthIncludingTerminator) {
  IsaNameLengthIncludesTerminator = true;
  activate();

  hsa_isa_t Isa{};
  EXPECT_EQ(Core.hsa_agent_get_info_fn(GpuAgent, HSA_AGENT_INFO_ISA, &Isa),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Isa.handle, PresentedIsa.handle);
}

TEST_F(HotswapHsaToolApiTest, RejectsUnsupportedPresentedWavefrontSize) {
  PresentedWavefrontSize = 16;
  ASSERT_EQ(setenv("HSA_HOTSWAP_PRESENT_ISA", "gfx1250", 1), 0);
  EXPECT_FALSE(OnLoad(&Root, HSA_API_TABLE_MAJOR_VERSION, 0, nullptr));
}

TEST_F(HotswapHsaToolApiTest, CapturesEveryReaderCreationRoute) {
  activate();
  char Path[] = "/tmp/comgr-hotswap-reader-XXXXXX";
  const int File = mkstemp(Path);
  ASSERT_NE(File, -1);
  ASSERT_EQ(unlink(Path), 0);
  constexpr char Contents[] = "not an ELF";
  ASSERT_EQ(pwrite(File, Contents, sizeof(Contents) - 1, 0),
            static_cast<ssize_t>(sizeof(Contents) - 1));

  hsa_code_object_reader_t MemoryReader{};
  EXPECT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Contents, sizeof(Contents) - 1, &MemoryReader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(ReaderMemoryCalls, 1u);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                hsa_executable_t{77}, GpuAgent, MemoryReader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);

  hsa_code_object_reader_t FileReader{};
  EXPECT_EQ(Core.hsa_code_object_reader_create_from_file_fn(File, &FileReader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(ReaderFileCalls, 1u);
  EXPECT_EQ(ReaderMemoryCalls, 1u);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                hsa_executable_t{77}, GpuAgent, FileReader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);

  hsa_ven_amd_loader_1_03_pfn_t Loader{};
  EXPECT_EQ(Core.hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1, sizeof(Loader), &Loader),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(
      Loader
          .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
      fakeLoaderReaderFileSlice);
  hsa_code_object_reader_t SliceReader{};
  EXPECT_EQ(
      Loader
          .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
              File, 0, sizeof(Contents) - 1, &SliceReader),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(ReaderFileSliceCalls, 1u);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                hsa_executable_t{77}, GpuAgent, SliceReader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);

  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(MemoryReader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(FileReader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(SliceReader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(close(File), 0);
}

TEST_F(HotswapHsaToolApiTest, RejectsUnaddressableMemoryReaderSize) {
  activate();
  hsa_code_object_reader_t Reader{};
  const size_t Unaddressable =
      static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) + 1;
  EXPECT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                reinterpret_cast<const void *>(1), Unaddressable, &Reader),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(ReaderMemoryCalls, 0u);
  EXPECT_EQ(Reader.handle, 0u);
}

TEST_F(HotswapHsaToolApiTest, PatchesEachKnownLoaderTableType) {
  activate();

  hsa_ven_amd_loader_1_00_pfn_t Table100{};
  ASSERT_EQ(Core.hsa_system_get_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                   0, &Table100),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(Table100.hsa_ven_amd_loader_query_host_address,
            fakeLoaderHostAddress);
  EXPECT_NE(Table100.hsa_ven_amd_loader_query_segment_descriptors,
            fakeLoaderQuerySegments);
  EXPECT_NE(Table100.hsa_ven_amd_loader_query_executable,
            fakeLoaderQueryExecutable);

  hsa_ven_amd_loader_1_01_pfn_t Table101{};
  ASSERT_EQ(Core.hsa_system_get_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                   1, &Table101),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(Table101.hsa_ven_amd_loader_executable_iterate_loaded_code_objects,
            fakeLoaderIterateLoaded);

  hsa_ven_amd_loader_1_02_pfn_t Table102{};
  ASSERT_EQ(Core.hsa_system_get_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                   2, &Table102),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(
      Table102
          .hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size,
      fakeLoaderReaderFileSlice);

  hsa_ven_amd_loader_1_03_pfn_t Table103{};
  ASSERT_EQ(Core.hsa_system_get_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                   3, &Table103),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(Table103.hsa_ven_amd_loader_iterate_executables,
            fakeLoaderIterateExecutables);
}

TEST_F(HotswapHsaToolApiTest, SerializesConcurrentLoaderTablePatching) {
  activate();
  constexpr size_t ThreadCount = 16;
  std::array<hsa_ven_amd_loader_1_03_pfn_t, ThreadCount> Tables{};
  std::array<hsa_status_t, ThreadCount> Statuses{};
  std::array<std::thread, ThreadCount> Threads;
  for (size_t I = 0; I != ThreadCount; ++I)
    Threads[I] = std::thread([&, I] {
      Statuses[I] = Core.hsa_system_get_major_extension_table_fn(
          HSA_EXTENSION_AMD_LOADER, 1, sizeof(Tables[I]), &Tables[I]);
    });
  for (std::thread &Thread : Threads)
    Thread.join();

  for (size_t I = 0; I != ThreadCount; ++I) {
    EXPECT_EQ(Statuses[I], HSA_STATUS_SUCCESS);
    EXPECT_NE(Tables[I].hsa_ven_amd_loader_query_host_address,
              fakeLoaderHostAddress);
    EXPECT_NE(Tables[I].hsa_ven_amd_loader_iterate_executables,
              fakeLoaderIterateExecutables);
  }
}

TEST_F(HotswapHsaToolApiTest,
       PatchesOnlyCompleteSlotsInArbitrarySizeLoaderTable) {
  activate();
  using LoaderTable = hsa_ven_amd_loader_1_03_pfn_t;
  const auto CheckSlot = [&](auto Original, size_t Offset) {
    using Function = decltype(Original);
    const size_t End = Offset + sizeof(Function);
    for (const size_t Size : {End - 1, End}) {
      std::vector<unsigned char> Storage(sizeof(LoaderTable) + 16, 0xa5);
      ASSERT_EQ(Core.hsa_system_get_major_extension_table_fn(
                    HSA_EXTENSION_AMD_LOADER, 1, Size, Storage.data()),
                HSA_STATUS_SUCCESS);
      for (size_t I = Size; I != Storage.size(); ++I)
        EXPECT_EQ(Storage[I], 0xa5) << "size " << Size << ", byte " << I;
      if (Size == End) {
        Function Patched;
        llvm::ArrayRef<unsigned char> Bytes(Storage);
        Bytes = Bytes.drop_front(Offset).take_front(sizeof(Patched));
        std::memcpy(&Patched, Bytes.data(), sizeof(Patched));
        EXPECT_NE(Patched, Original);
      }
    }
  };

  CheckSlot(fakeLoaderHostAddress,
            offsetof(LoaderTable, hsa_ven_amd_loader_query_host_address));
  CheckSlot(
      fakeLoaderQuerySegments,
      offsetof(LoaderTable, hsa_ven_amd_loader_query_segment_descriptors));
  CheckSlot(fakeLoaderQueryExecutable,
            offsetof(LoaderTable, hsa_ven_amd_loader_query_executable));
  CheckSlot(
      fakeLoaderIterateLoaded,
      offsetof(LoaderTable,
               hsa_ven_amd_loader_executable_iterate_loaded_code_objects));
  CheckSlot(
      fakeLoaderReaderFileSlice,
      offsetof(
          LoaderTable,
          hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size));
  CheckSlot(fakeLoaderIterateExecutables,
            offsetof(LoaderTable, hsa_ven_amd_loader_iterate_executables));
}

TEST_F(HotswapHsaToolApiTest, RejectsNullLoaderExtensionTableOutput) {
  activate();
  AllowNullLoaderTableSuccess = true;
  EXPECT_EQ(Core.hsa_system_get_extension_table_fn(HSA_EXTENSION_AMD_LOADER, 1,
                                                   3, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(Core.hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1,
                sizeof(hsa_ven_amd_loader_1_03_pfn_t), nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F(HotswapHsaToolApiTest, FailsWhenExecutionIsaQueryIsUnavailable) {
  ASSERT_EQ(setenv("HSA_HOTSWAP_PRESENT_ISA", "gfx1250", 1), 0);
  FailExecutionIsa = true;
  auto *AgentGetInfo = Core.hsa_agent_get_info_fn;
  EXPECT_FALSE(OnLoad(&Root, HSA_API_TABLE_MAJOR_VERSION, 0, nullptr));
  EXPECT_EQ(Core.hsa_agent_get_info_fn, AgentGetInfo);
}

TEST_F(HotswapHsaToolApiTest, ProtectsCoreAndInterceptQueueCreation) {
  activate();
  hsa_queue_t *Created = nullptr;
  EXPECT_EQ(Core.hsa_queue_create_fn(GpuAgent, 64, HSA_QUEUE_TYPE_MULTI,
                                     nullptr, nullptr, 0, UINT32_MAX, &Created),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Created, &Queue);
  EXPECT_EQ(NativeQueueCreateCalls, 0u);
  EXPECT_EQ(InterceptQueueCreateCalls, 1u);
  EXPECT_EQ(InterceptRegisterCalls, 1u);
  EXPECT_NE(RegisteredInterceptor, nullptr);

  uint64_t Doorbell = 0;
  EXPECT_EQ(Amd.hsa_amd_queue_get_info_fn(
                Created, HSA_AMD_QUEUE_INFO_DOORBELL_ID, &Doorbell),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(QueueGetInfoCalls, 0u);
  EXPECT_EQ(
      Amd.hsa_amd_queue_set_priority_fn(Created, HSA_AMD_QUEUE_PRIORITY_HIGH),
      HSA_STATUS_SUCCESS);
  uint32_t Mask = 1;
  EXPECT_EQ(Amd.hsa_amd_queue_cu_set_mask_fn(Created, 32, &Mask),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(PriorityCalls, 1u);
  EXPECT_EQ(CuMaskCalls, 1u);
  EXPECT_EQ(Core.hsa_queue_destroy_fn(Created), HSA_STATUS_SUCCESS);

  Created = nullptr;
  EXPECT_EQ(Amd.hsa_amd_queue_intercept_create_fn(
                GpuAgent, 64, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0,
                UINT32_MAX, &Created),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Created, &Queue);
  EXPECT_EQ(InterceptQueueCreateCalls, 2u);
  EXPECT_EQ(InterceptRegisterCalls, 2u);
}

TEST_F(HotswapHsaToolApiTest, RefusesUnsupportedQueueType) {
  activate();
  hsa_queue_t *Created = nullptr;
  EXPECT_EQ(Core.hsa_queue_create_fn(GpuAgent, 64, HSA_QUEUE_TYPE_SINGLE,
                                     nullptr, nullptr, 0, UINT32_MAX, &Created),
            HSA_STATUS_ERROR_INVALID_QUEUE_CREATION);
  EXPECT_EQ(Created, nullptr);
  EXPECT_EQ(InterceptQueueCreateCalls, 0u);
}

TEST_F(HotswapHsaToolApiTest, RejectsNullProtectedQueueOutput) {
  activate();
  EXPECT_EQ(Core.hsa_queue_create_fn(GpuAgent, 64, HSA_QUEUE_TYPE_MULTI,
                                     nullptr, nullptr, 0, UINT32_MAX, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(InterceptQueueCreateCalls, 0u);
}

TEST_F(HotswapHsaToolApiTest, RefusesUnprotectedSoftQueueCreation) {
  activate();
  hsa_queue_t *Created = nullptr;
  EXPECT_EQ(Core.hsa_soft_queue_create_fn(hsa_region_t{1}, 64,
                                          HSA_QUEUE_TYPE_MULTI, 1,
                                          hsa_signal_t{1}, &Created),
            HSA_STATUS_ERROR_NOT_SUPPORTED);
  EXPECT_EQ(Created, nullptr);
  EXPECT_EQ(SoftQueueCreateCalls, 0u);
}

TEST_F(HotswapHsaToolApiTest, ProtectsAmdBatchQueueAndAppliesPolicy) {
  activate();
  uint32_t Mask = 1;
  hsa_amd_queue_create_desc_t Descriptor{};
  Descriptor.version = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
  Descriptor.queue_size_bytes = 4 * sizeof(hsa_kernel_dispatch_packet_t);
  Descriptor.priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
  Descriptor.engine_type = HSA_AMD_QUEUE_ENGINE_COMPUTE;
  Descriptor.engine.compute.type = HSA_QUEUE_TYPE_MULTI;
  Descriptor.engine.compute.cu_mask_count = 32;
  Descriptor.engine.compute.cu_mask = &Mask;

  EXPECT_EQ(Amd.hsa_amd_queue_create_fn(GpuAgent, &Descriptor, 1),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Descriptor.queue, &Queue);
  EXPECT_EQ(InterceptQueueCreateCalls, 1u);
  EXPECT_EQ(InterceptRegisterCalls, 1u);
  EXPECT_EQ(PriorityCalls, 1u);
  EXPECT_EQ(CuMaskCalls, 1u);
}

TEST_F(HotswapHsaToolApiTest, PassesBarrierPacketsWithoutRewriting) {
  activate();
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
  RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                        fakePacketWriter);
  EXPECT_EQ(WriterCalls, 1u);
  EXPECT_EQ(WrittenPacketCount, 1u);
  EXPECT_EQ(WrittenPacket.header, Packet.header);
}

TEST_F(HotswapHsaToolApiTest, RejectsPacketBatchLargerThanQueueCapacity) {
  ReturnedQueueSize = 4;
  activate();
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 5, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "invalid queue-intercept callback arguments");
  EXPECT_EQ(WriterCalls, 0u);
}

TEST_F(HotswapHsaToolApiTest, RefusesUnregisteredKernelDispatch) {
  char Path[] = "/tmp/comgr-hotswap-rejected-dispatch-XXXXXX";
  const int File = mkstemp(Path);
  ASSERT_NE(File, -1);
  ASSERT_EQ(close(File), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_PROOF_LOG", Path, 1), 0);
  activate();
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  Packet.kernel_object = 0x1234;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "unregistered kernel_object");
  EXPECT_EQ(WriterCalls, 0u);
  EXPECT_EQ(Core.hsa_queue_destroy_fn(&Queue), HSA_STATUS_SUCCESS);
  OnUnload();

  const std::string Contents = readFile(Path);
  EXPECT_EQ(unlink(Path), 0);
  EXPECT_NE(Contents.find("\"event\":\"dispatch_intercepted\""),
            std::string::npos);
  EXPECT_NE(Contents.find("\"event\":\"dispatch_rejected\""),
            std::string::npos);
}

TEST_F(HotswapHsaToolApiTest, RefusesMalformedDispatchEncoding) {
  activate();
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "invalid dimension count");

  Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.reserved0 = 1;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "nonzero reserved field");

  Packet.reserved0 = 0;
  Packet.workgroup_size_y = 2;
  Packet.grid_size_y = 2;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "non-unit inactive dimensions");

  Packet.header |= uint16_t{1} << 15;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "nonzero reserved bits");
}

TEST_F(HotswapHsaToolApiTest, RefusesUnsupportedPacketAndVendorFormats) {
  activate();
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_AGENT_DISPATCH << HSA_PACKET_HEADER_TYPE;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "unsupported AQL packet type");

  amd_aql_intercept_marker_t Marker{};
  static_assert(sizeof(Marker) == sizeof(Packet));
  Marker.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  Marker.format = AMD_AQL_FORMAT_INTERCEPT_MARKER;
  std::memcpy(&Packet, &Marker, sizeof(Packet));
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "unsupported vendor packet format");

  hsa_amd_ext_kernel_dispatch_packet_t Extended{};
  Extended.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  Extended.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
  Extended.workgroup_size_x = 1;
  Extended.workgroup_size_y = 1;
  Extended.workgroup_size_z = 1;
  Extended.cluster_count_x = 1;
  Extended.cluster_count_y = 1;
  Extended.cluster_count_z = 1;
  Extended.cluster_size_x = 1;
  Extended.cluster_size_y = 1;
  Extended.cluster_size_z = 1;
  Extended.perf_hint.hint_val = 1;
  std::memcpy(&Packet, &Extended, sizeof(Packet));
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "unsupported reserved, dependency, or performance-hint");
  EXPECT_EQ(WriterCalls, 0u);
}

TEST_F(HotswapHsaToolApiTest, RestoresApiTablesOnUnloadAndCanReload) {
  auto *AgentGetInfo = Core.hsa_agent_get_info_fn;
  auto *QueueCreate = Core.hsa_queue_create_fn;
  auto *AmdQueueCreate = Amd.hsa_amd_queue_create_fn;
  activate();
  EXPECT_NE(Core.hsa_agent_get_info_fn, AgentGetInfo);
  EXPECT_NE(Core.hsa_queue_create_fn, QueueCreate);
  EXPECT_NE(Amd.hsa_amd_queue_create_fn, AmdQueueCreate);

  OnUnload();
  EXPECT_EQ(Core.hsa_agent_get_info_fn, AgentGetInfo);
  EXPECT_EQ(Core.hsa_queue_create_fn, QueueCreate);
  EXPECT_EQ(Amd.hsa_amd_queue_create_fn, AmdQueueCreate);
  EXPECT_TRUE(OnLoad(&Root, HSA_API_TABLE_MAJOR_VERSION, 0, nullptr));
}

TEST_F(HotswapHsaToolApiTest, DoesNotClobberLaterWrapperAtUnload) {
  activate();
  Core.hsa_agent_get_info_fn = outerAgentGetInfo;

  OnUnload();
  EXPECT_EQ(Core.hsa_agent_get_info_fn, outerAgentGetInfo);
}

TEST_F(HotswapHsaToolApiTest, ProofSummaryMatchesProtectedQueueEvents) {
  char Path[] = "/tmp/comgr-hotswap-proof-XXXXXX";
  const int File = mkstemp(Path);
  ASSERT_NE(File, -1);
  ASSERT_EQ(close(File), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_PROOF_LOG", Path, 1), 0);
  activate();
  createProtectedQueue();
  ASSERT_EQ(Core.hsa_queue_destroy_fn(&Queue), HSA_STATUS_SUCCESS);
  OnUnload();

  const std::string Contents = readFile(Path);
  EXPECT_EQ(unlink(Path), 0);
  EXPECT_NE(Contents.find("\"pid\":"), std::string::npos);
  EXPECT_NE(Contents.find("\"event\":\"queue_protected\""), std::string::npos);
  EXPECT_NE(Contents.find("\"protected_queues_created\":1"), std::string::npos);
  EXPECT_NE(Contents.find("\"intercepted_dispatches\":0"), std::string::npos);
  EXPECT_NE(Contents.find("\"rewritten_dispatches\":0"), std::string::npos);
  EXPECT_NE(Contents.find("\"all_intercepted_dispatches_rewritten\":true"),
            std::string::npos);
  EXPECT_NE(Contents.find("\"event\":\"tool_unloaded\""), std::string::npos);
}

TEST_F(HotswapHsaToolApiTest, RejectsUnloadWithLiveProtectedQueue) {
  activate();
  createProtectedQueue();
  EXPECT_DEATH(OnUnload(), "live protected queues");
  EXPECT_EQ(Core.hsa_queue_destroy_fn(&Queue), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest, FailedNativeDestroyPreservesMixingRefusal) {
  EnableNativeProgramLoad = true;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  const hsa_executable_t Executable{77};
  ASSERT_EQ(Core.hsa_executable_load_program_code_object_fn(Executable, Reader,
                                                            nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ExecutableDestroyResult = HSA_STATUS_ERROR_INVALID_EXECUTABLE;
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable),
            HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Executable, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest,
       ExecutableDestroyCannotCrossInFlightNativeMutation) {
  EnableNativeProgramLoad = true;
  BlockNativeProgramLoad = true;
  activate("gfx950");

  const hsa_executable_t Executable{77};
  const hsa_code_object_reader_t Reader{88};
  hsa_status_t LoadStatus = HSA_STATUS_ERROR;
  std::thread Loader([&] {
    LoadStatus = Core.hsa_executable_load_program_code_object_fn(
        Executable, Reader, nullptr, nullptr);
  });
  {
    std::unique_lock<std::mutex> Lock(MutationMutex);
    MutationCondition.wait(Lock, [] { return NativeProgramLoadEntered; });
  }

  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable),
            HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  EXPECT_EQ(ExecutableDestroyCalls, 0u);
  {
    std::lock_guard<std::mutex> Lock(MutationMutex);
    ReleaseNativeProgramLoad = true;
  }
  MutationCondition.notify_all();
  Loader.join();

  EXPECT_EQ(LoadStatus, HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 1u);
}

TEST_F(HotswapHsaToolApiTest, ExecutableDestroyCannotCrossSourceRegistration) {
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  BlockExecutableGetInfo = true;
  const hsa_executable_t Executable{77};
  hsa_status_t LoadStatus = HSA_STATUS_ERROR;
  std::thread Loader([&] {
    LoadStatus = Core.hsa_executable_load_agent_code_object_fn(
        Executable, GpuAgent, Reader, nullptr, nullptr);
  });
  {
    std::unique_lock<std::mutex> Lock(MutationMutex);
    MutationCondition.wait(Lock, [] { return ExecutableGetInfoEntered; });
  }

  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable),
            HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  EXPECT_EQ(ExecutableDestroyCalls, 0u);
  {
    std::lock_guard<std::mutex> Lock(MutationMutex);
    ReleaseExecutableGetInfo = true;
  }
  MutationCondition.notify_all();
  Loader.join();

  EXPECT_EQ(LoadStatus, HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 1u);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest, ExecutableMutationAndDestroyCannotCrossFreeze) {
  activate("gfx950");
  BlockExecutableFreeze = true;

  const hsa_executable_t Executable{77};
  hsa_status_t FreezeStatus = HSA_STATUS_ERROR;
  std::thread Freezer([&] {
    FreezeStatus = Core.hsa_executable_freeze_fn(Executable, nullptr);
  });
  {
    std::unique_lock<std::mutex> Lock(MutationMutex);
    MutationCondition.wait(Lock, [] { return ExecutableFreezeEntered; });
  }

  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable),
            HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  EXPECT_EQ(ExecutableDestroyCalls, 0u);
  EXPECT_EQ(Core.hsa_executable_load_program_code_object_fn(
                Executable, hsa_code_object_reader_t{88}, nullptr, nullptr),
            HSA_STATUS_ERROR_NOT_SUPPORTED);
  {
    std::lock_guard<std::mutex> Lock(MutationMutex);
    ReleaseExecutableFreeze = true;
  }
  MutationCondition.notify_all();
  Freezer.join();

  EXPECT_EQ(FreezeStatus, HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Executable), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 1u);
}

TEST_F(HotswapHsaToolApiTest,
       SuccessfulGlobalDefinitionPreventsSourceRegistration) {
  EnableGlobalDefinitions = true;
  activate("gfx950");

  int Storage = 0;
  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  const hsa_executable_t GlobalExecutable{77};
  ASSERT_EQ(Core.hsa_executable_global_variable_define_fn(
                GlobalExecutable, "host_defined", &Storage),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                GlobalExecutable, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);

  const hsa_executable_t AgentGlobalExecutable{78};
  ASSERT_EQ(Core.hsa_executable_agent_global_variable_define_fn(
                AgentGlobalExecutable, GpuAgent, "agent_defined", &Storage),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                AgentGlobalExecutable, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);

  const hsa_executable_t ReadonlyExecutable{79};
  ASSERT_EQ(Core.hsa_executable_readonly_variable_define_fn(
                ReadonlyExecutable, GpuAgent, "readonly_defined", &Storage),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                ReadonlyExecutable, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest, RejectsSourceObjectWithSharedDeviceStorage) {
  activate("gfx950");

  const int ObjectFile =
      open(COMGR_HOTSWAP_HSA_TOOL_GLOBAL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                hsa_executable_t{77}, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_ERROR_INVALID_CODE_OBJECT);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST(HotswapHsaToolObject, RejectsInvalidStorageAndKernelLinkage) {
  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);

  const std::array<std::string, 1> KernelDescriptors{"vecadd.kd"};
  std::string Failure;
  ASSERT_TRUE(COMGR::hotswap::hsa_tool::inspectSourceStorage(
      *Object, KernelDescriptors, Failure));

  const llvm::StringRef Bytes(reinterpret_cast<const char *>(Object->data()),
                              Object->size());
  auto Parsed = llvm::object::ObjectFile::createObjectFile(
      llvm::MemoryBufferRef(Bytes, "source-object"));
  if (!Parsed)
    FAIL() << llvm::toString(Parsed.takeError());
  const auto *Elf =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(Parsed->get());
  ASSERT_NE(Elf, nullptr);

  std::vector<uint8_t> LocalBinding = *Object;
  bool ChangedBinding = false;
  for (const llvm::object::SymbolRef &Symbol : Elf->symbols()) {
    auto Name = Symbol.getName();
    if (!Name)
      FAIL() << llvm::toString(Name.takeError());
    if (*Name != "vecadd.kd")
      continue;
    auto RawSymbol = Elf->getSymbol(Symbol.getRawDataRefImpl());
    if (!RawSymbol)
      FAIL() << llvm::toString(RawSymbol.takeError());
    const uintptr_t Base = reinterpret_cast<uintptr_t>(Object->data());
    const uintptr_t SymbolAddress = reinterpret_cast<uintptr_t>(*RawSymbol);
    ASSERT_GE(SymbolAddress, Base);
    const size_t Offset = static_cast<size_t>(SymbolAddress - Base);
    ASSERT_LE(Offset, LocalBinding.size());
    ASSERT_LE(sizeof(**RawSymbol), LocalBinding.size() - Offset);
    auto LocalSymbol = **RawSymbol;
    LocalSymbol.setBinding(llvm::ELF::STB_LOCAL);
    llvm::MutableArrayRef<uint8_t> Bytes(LocalBinding);
    Bytes = Bytes.drop_front(Offset).take_front(sizeof(LocalSymbol));
    std::memcpy(Bytes.data(), &LocalSymbol, sizeof(LocalSymbol));
    ChangedBinding = true;
    break;
  }
  ASSERT_TRUE(ChangedBinding);
  Failure.clear();
  EXPECT_FALSE(COMGR::hotswap::hsa_tool::inspectSourceStorage(
      LocalBinding, KernelDescriptors, Failure));
  EXPECT_NE(Failure.find("does not have global linkage"), std::string::npos);

  auto Headers = Elf->getELFFile().program_headers();
  if (!Headers)
    FAIL() << llvm::toString(Headers.takeError());

  std::vector<uint8_t> Modified = *Object;
  bool Changed = false;
  for (const auto &Header : *Headers) {
    if (Header.p_type != llvm::ELF::PT_LOAD ||
        !(Header.p_flags & llvm::ELF::PF_W))
      continue;
    const uintptr_t Base = reinterpret_cast<uintptr_t>(Object->data());
    const uintptr_t HeaderAddress = reinterpret_cast<uintptr_t>(&Header);
    ASSERT_GE(HeaderAddress, Base);
    const size_t Offset = static_cast<size_t>(HeaderAddress - Base);
    ASSERT_LE(Offset, Modified.size());
    ASSERT_LE(sizeof(Header), Modified.size() - Offset);
    ASSERT_LT(static_cast<uint64_t>(Header.p_memsz),
              std::numeric_limits<uint64_t>::max());
    auto Enlarged = Header;
    Enlarged.p_memsz = static_cast<uint64_t>(Header.p_memsz) + 1;
    llvm::MutableArrayRef<uint8_t> Bytes(Modified);
    Bytes = Bytes.drop_front(Offset).take_front(sizeof(Enlarged));
    std::memcpy(Bytes.data(), &Enlarged, sizeof(Enlarged));
    Changed = true;
    break;
  }
  ASSERT_TRUE(Changed);

  Failure.clear();
  EXPECT_FALSE(COMGR::hotswap::hsa_tool::inspectSourceStorage(
      Modified, KernelDescriptors, Failure));
  EXPECT_NE(Failure.find("writable load segment contains unsupported storage"),
            std::string::npos);

  Modified = *Object;
  Changed = false;
  for (const auto &Header : *Headers) {
    if (Header.p_type != llvm::ELF::PT_GNU_RELRO || Header.p_memsz == 0)
      continue;
    const uintptr_t Base = reinterpret_cast<uintptr_t>(Object->data());
    const uintptr_t HeaderAddress = reinterpret_cast<uintptr_t>(&Header);
    ASSERT_GE(HeaderAddress, Base);
    const size_t Offset = static_cast<size_t>(HeaderAddress - Base);
    ASSERT_LE(Offset, Modified.size());
    ASSERT_LE(sizeof(Header), Modified.size() - Offset);
    auto Shortened = Header;
    Shortened.p_memsz = static_cast<uint64_t>(Header.p_memsz) - 1;
    llvm::MutableArrayRef<uint8_t> Bytes(Modified);
    Bytes = Bytes.drop_front(Offset).take_front(sizeof(Shortened));
    std::memcpy(Bytes.data(), &Shortened, sizeof(Shortened));
    Changed = true;
    break;
  }
  ASSERT_TRUE(Changed);

  Failure.clear();
  EXPECT_FALSE(COMGR::hotswap::hsa_tool::inspectSourceStorage(
      Modified, KernelDescriptors, Failure));
  EXPECT_NE(Failure.find(".relro_padding section is outside GNU_RELRO"),
            std::string::npos);
}

TEST_F(HotswapHsaToolApiTest, AcceptsStrippedObjectWhoseBssIsOnlyTheHipMarker) {
  activate("gfx950");

  const int ObjectFile =
      open(COMGR_HOTSWAP_HSA_TOOL_STRIPPED_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);
  const hsa_executable_t Parent{77};
  EXPECT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Parent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest,
       LoaderHostQueryDoesNotForceTranslationOfVirtualKernel) {
  EnableTranslatedLoader = true;
  ExecutableCreateResult = HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  const hsa_executable_t Parent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr), HSA_STATUS_SUCCESS);
  hsa_executable_symbol_t Symbol{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(Parent, "vecadd",
                                                      &GpuAgent, &Symbol),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(Symbol.handle, 0u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);

  uint64_t KernelObject = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject),
      HSA_STATUS_SUCCESS);
  EXPECT_NE(KernelObject, 0u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  hsa_ven_amd_loader_1_03_pfn_t Loader{};
  ASSERT_EQ(Core.hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1, sizeof(Loader), &Loader),
            HSA_STATUS_SUCCESS);
  const void *HostAddress = nullptr;
  EXPECT_EQ(Loader.hsa_ven_amd_loader_query_host_address(
                reinterpret_cast<const void *>(KernelObject), &HostAddress),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(HostAddress, nullptr);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);

  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  Packet.kernel_object = KernelObject;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "translation of vecadd failed");
  EXPECT_EQ(WriterCalls, 0u);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Parent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest, RejectsIncompatibleTranslatedKernelAbi) {
  ASSERT_EQ(setenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO", "1", 1), 0);
  EnableTranslatedLoader = true;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  hsa_ven_amd_loader_1_03_pfn_t Loader{};
  ASSERT_EQ(Core.hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1, sizeof(Loader), &Loader),
            HSA_STATUS_SUCCESS);
  TranslatedDynamicCallstack = true;
  const hsa_executable_t Parent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr), HSA_STATUS_SUCCESS);
  hsa_executable_symbol_t Symbol{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(Parent, "vecadd",
                                                      &GpuAgent, &Symbol),
            HSA_STATUS_SUCCESS);
  uint64_t KernelObject = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject),
      HSA_STATUS_SUCCESS);
  const void *HostAddress = nullptr;
  EXPECT_EQ(Loader.hsa_ven_amd_loader_query_host_address(
                reinterpret_cast<const void *>(KernelObject), &HostAddress),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(HostAddress, nullptr);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);

  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  Packet.kernel_object = KernelObject;
  EXPECT_DEATH(RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                                     fakePacketWriter),
               "translated kernel unexpectedly uses a dynamic call stack");
  EXPECT_EQ(WriterCalls, 0u);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(Parent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest,
       TranslatesOnceAtFirstDispatchAfterLazySymbolOperations) {
  char ProofPath[] = "/tmp/comgr-hotswap-lifecycle-XXXXXX";
  const int ProofFile = mkstemp(ProofPath);
  ASSERT_NE(ProofFile, -1);
  ASSERT_EQ(close(ProofFile), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_PROOF_LOG", ProofPath, 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO", "1", 1), 0);
  EnableTranslatedLoader = true;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);

  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);
  const hsa_executable_t Parent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);

  unsigned Iterated = 0;
  EXPECT_EQ(Core.hsa_executable_iterate_agent_symbols_fn(
                Parent, GpuAgent, countAgentSymbol, &Iterated),
            HSA_STATUS_ERROR_INVALID_EXECUTABLE);
  EXPECT_EQ(Iterated, 0u);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr), HSA_STATUS_SUCCESS);

  hsa_executable_symbol_t InvalidSymbol{};
  EXPECT_EQ(Core.hsa_executable_get_symbol_fn(Parent, nullptr, "vecadd",
                                              GpuAgent, 0, nullptr),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);
  EXPECT_EQ(Core.hsa_executable_get_symbol_fn(Parent, nullptr, nullptr,
                                              GpuAgent, 0, &InvalidSymbol),
            HSA_STATUS_ERROR_INVALID_ARGUMENT);

  uint32_t ValidationResult = 1;
  ASSERT_EQ(Core.hsa_executable_validate_fn(Parent, &ValidationResult),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(ValidationResult, 0u);
  ValidationResult = 1;
  ASSERT_EQ(
      Core.hsa_executable_validate_alt_fn(Parent, nullptr, &ValidationResult),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(ValidationResult, 0u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);

  hsa_status_t FirstStatus = HSA_STATUS_ERROR;
  hsa_status_t SecondStatus = HSA_STATUS_ERROR;
  hsa_executable_symbol_t FirstSymbol{};
  hsa_executable_symbol_t SecondSymbol{};
  std::thread First([&] {
    FirstStatus = Core.hsa_executable_get_symbol_by_name_fn(
        Parent, "vecadd", &GpuAgent, &FirstSymbol);
  });
  std::thread Second([&] {
    SecondStatus = Core.hsa_executable_get_symbol_by_name_fn(
        Parent, "vecadd", &GpuAgent, &SecondSymbol);
  });
  First.join();
  Second.join();
  ASSERT_EQ(FirstStatus, HSA_STATUS_SUCCESS);
  ASSERT_EQ(SecondStatus, HSA_STATUS_SUCCESS);
  EXPECT_EQ(FirstSymbol.handle, SecondSymbol.handle);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);
  EXPECT_EQ(ReaderDestroyCalls, 0u);

  Iterated = 0;
  EXPECT_EQ(
      Core.hsa_executable_iterate_symbols_fn(Parent, countSymbol, &Iterated),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(Iterated, 1u);
  Iterated = 0;
  EXPECT_EQ(Core.hsa_executable_iterate_agent_symbols_fn(
                Parent, GpuAgent, countAgentSymbol, &Iterated),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Iterated, 1u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  hsa_symbol_kind_t Kind = HSA_SYMBOL_KIND_VARIABLE;
  uint32_t NameLength = 0;
  hsa_symbol_linkage_t Linkage = HSA_SYMBOL_LINKAGE_MODULE;
  bool IsDefinition = false;
  bool DynamicCallstack = true;
  uint32_t WavefrontSize = 0;
  hsa_agent_t SymbolAgent{};
  ASSERT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &Kind),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &NameLength),
      HSA_STATUS_SUCCESS);
  std::string SymbolName(NameLength, '\0');
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, SymbolName.data()),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE, &Linkage),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION, &IsDefinition),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_AGENT, &SymbolAgent),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK,
                &DynamicCallstack),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol,
                static_cast<hsa_executable_symbol_info_t>(
                    HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE),
                &WavefrontSize),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Kind, HSA_SYMBOL_KIND_KERNEL);
  EXPECT_EQ(SymbolName, "vecadd.kd");
  EXPECT_EQ(Linkage, HSA_SYMBOL_LINKAGE_PROGRAM);
  EXPECT_TRUE(IsDefinition);
  EXPECT_EQ(SymbolAgent.handle, GpuAgent.handle);
  EXPECT_FALSE(DynamicCallstack);
  EXPECT_EQ(WavefrontSize, 64u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  uint64_t KernelObject = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject),
      HSA_STATUS_SUCCESS);
  EXPECT_NE(KernelObject, 0u);
  EXPECT_NE(KernelObject, 0x1234000u);
  uint32_t KernargSize = 0;
  uint32_t KernargAlignment = 0;
  EXPECT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &KernargSize),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_symbol_get_info_fn(
                FirstSymbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                &KernargAlignment),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(KernargSize, 288u);
  EXPECT_EQ(KernargAlignment, 16u);

  hsa_ven_amd_loader_1_03_pfn_t Loader{};
  ASSERT_EQ(Core.hsa_system_get_major_extension_table_fn(
                HSA_EXTENSION_AMD_LOADER, 1, sizeof(Loader), &Loader),
            HSA_STATUS_SUCCESS);
  hsa_executable_t TokenExecutable{};
  EXPECT_EQ(Loader.hsa_ven_amd_loader_query_executable(
                reinterpret_cast<const void *>(KernelObject), &TokenExecutable),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(TokenExecutable.handle, Parent.handle);
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  Packet.kernel_object = KernelObject;
  RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                        fakePacketWriter);
  ASSERT_EQ(WriterCalls, 1u);
  EXPECT_EQ(WrittenPacket.kernel_object, 0x1234000u);
  EXPECT_EQ(ExecutableCreateCalls, 1u);
  EXPECT_EQ(TranslatedLoadCalls, 1u);
  EXPECT_EQ(ReaderDestroyCalls, 1u);

  const void *HostAddress = nullptr;
  EXPECT_EQ(Loader.hsa_ven_amd_loader_query_host_address(
                reinterpret_cast<const void *>(KernelObject), &HostAddress),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(HostAddress, reinterpret_cast<const void *>(0x5678000));
  RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                        fakePacketWriter);
  ASSERT_EQ(WriterCalls, 2u);
  EXPECT_EQ(WrittenPacket.kernel_object, 0x1234000u);

  hsa_amd_ext_kernel_dispatch_packet_t Extended{};
  Extended.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  Extended.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
  Extended.setup = 2 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Extended.workgroup_size_x = 2;
  Extended.workgroup_size_y = 3;
  Extended.workgroup_size_z = 1;
  Extended.cluster_count_x = 4;
  Extended.cluster_count_y = 5;
  Extended.cluster_count_z = 1;
  Extended.cluster_size_x = 1;
  Extended.cluster_size_y = 1;
  Extended.cluster_size_z = 1;
  Extended.kernel_object = KernelObject;
  hsa_kernel_dispatch_packet_t ExtendedStorage{};
  static_assert(sizeof(Extended) == sizeof(ExtendedStorage));
  std::memcpy(&ExtendedStorage, &Extended, sizeof(Extended));
  RegisteredInterceptor(&ExtendedStorage, 1, 0, RegisteredInterceptorData,
                        fakePacketWriter);
  ASSERT_EQ(WriterCalls, 3u);
  EXPECT_EQ((WrittenPacket.header >> HSA_PACKET_HEADER_TYPE) &
                ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1),
            HSA_PACKET_TYPE_KERNEL_DISPATCH);
  EXPECT_EQ(WrittenPacket.kernel_object, 0x1234000u);
  EXPECT_EQ(WrittenPacket.grid_size_x, 8u);
  EXPECT_EQ(WrittenPacket.grid_size_y, 15u);
  EXPECT_EQ(WrittenPacket.grid_size_z, 1u);
  EXPECT_EQ(ExecutableCreateCalls, 1u);

  EXPECT_EQ(Core.hsa_executable_destroy_fn(Parent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 2u);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ReaderDestroyCalls, 2u);
  EXPECT_EQ(Core.hsa_queue_destroy_fn(&Queue), HSA_STATUS_SUCCESS);
  OnUnload();

  const std::string Contents = readFile(ProofPath);
  EXPECT_EQ(unlink(ProofPath), 0);
  EXPECT_NE(Contents.find("\"registered_source_objects\":1"),
            std::string::npos);
  EXPECT_NE(Contents.find("\"registered_kernels\":1"), std::string::npos);
  EXPECT_NE(Contents.find("\"translation_requests\":1"), std::string::npos);
  EXPECT_NE(Contents.find("\"successful_translations\":1"), std::string::npos);
  EXPECT_NE(Contents.find("\"intercepted_dispatches\":3"), std::string::npos);
  EXPECT_NE(Contents.find("\"rewritten_dispatches\":3"), std::string::npos);
  EXPECT_NE(Contents.find("\"event\":\"extended_dispatch_lowered\""),
            std::string::npos);
  EXPECT_NE(Contents.find("\"event\":\"dispatch_intercepted\""),
            std::string::npos);
  EXPECT_NE(Contents.find("\"all_intercepted_dispatches_rewritten\":true"),
            std::string::npos);
}

TEST_F(HotswapHsaToolApiTest,
       SymbolIterationKeepsVirtualSymbolAliveAcrossConcurrentParentDestroy) {
  ASSERT_EQ(setenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO", "1", 1), 0);
  EnableTranslatedLoader = true;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);
  const hsa_executable_t Parent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr), HSA_STATUS_SUCCESS);
  hsa_executable_symbol_t PreparedSymbol{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(
                Parent, "vecadd", &GpuAgent, &PreparedSymbol),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  IterationDestroyContext Context;
  Context.Core = &Core;
  hsa_status_t IterationStatus = HSA_STATUS_ERROR;
  std::thread Iteration([&] {
    IterationStatus = Core.hsa_executable_iterate_symbols_fn(
        Parent, querySymbolAfterParentDestroy, &Context);
  });
  {
    std::unique_lock<std::mutex> Lock(Context.Mutex);
    Context.Condition.wait(Lock, [&] { return Context.CallbackEntered; });
  }

  EXPECT_EQ(Core.hsa_executable_destroy_fn(Parent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 1u);
  {
    std::lock_guard<std::mutex> Lock(Context.Mutex);
    Context.ParentDestroyed = true;
  }
  Context.Condition.notify_all();
  Iteration.join();

  EXPECT_EQ(IterationStatus, HSA_STATUS_SUCCESS);
  EXPECT_EQ(Context.SymbolInfoStatus, HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 1u);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
}

TEST_F(HotswapHsaToolApiTest,
       DefersLiveChildToRuntimeLoaderWhenRuntimeUnloads) {
  char ProofPath[] = "/tmp/comgr-hotswap-runtime-teardown-XXXXXX";
  const int ProofFile = mkstemp(ProofPath);
  ASSERT_NE(ProofFile, -1);
  ASSERT_EQ(close(ProofFile), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_PROOF_LOG", ProofPath, 1), 0);
  ASSERT_EQ(setenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO", "1", 1), 0);
  EnableTranslatedLoader = true;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);

  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);
  const hsa_executable_t Parent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr), HSA_STATUS_SUCCESS);
  hsa_executable_symbol_t Symbol{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(Parent, "vecadd",
                                                      &GpuAgent, &Symbol),
            HSA_STATUS_SUCCESS);
  uint64_t KernelObject = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &KernelObject),
      HSA_STATUS_SUCCESS);
  createProtectedQueue();
  hsa_kernel_dispatch_packet_t Packet{};
  Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
  Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  Packet.workgroup_size_x = 1;
  Packet.workgroup_size_y = 1;
  Packet.workgroup_size_z = 1;
  Packet.grid_size_x = 1;
  Packet.grid_size_y = 1;
  Packet.grid_size_z = 1;
  Packet.kernel_object = KernelObject;
  RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                        fakePacketWriter);
  ASSERT_EQ(WriterCalls, 1u);
  EXPECT_EQ(ReaderDestroyCalls, 1u);
  EXPECT_EQ(ExecutableDestroyCalls, 0u);
  ASSERT_EQ(Core.hsa_queue_destroy_fn(&Queue), HSA_STATUS_SUCCESS);

  OnUnload();
  EXPECT_EQ(ExecutableDestroyCalls, 0u);

  const std::string Contents = readFile(ProofPath);
  EXPECT_EQ(unlink(ProofPath), 0);
  EXPECT_NE(Contents.find("\"event\":\"runtime_teardown_children\","
                          "\"count\":1"),
            std::string::npos);
  EXPECT_NE(Contents.find("\"event\":\"tool_unloaded\""), std::string::npos);
}

TEST_F(HotswapHsaToolApiTest,
       TranslatesOnlyDispatchedKernelsInAMultiKernelExecutable) {
  EnableTranslatedLoader = true;
  activate("gfx950");

  const int ObjectFile =
      open(COMGR_HOTSWAP_HSA_TOOL_TWO_KERNEL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  const hsa_executable_t Parent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                Parent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr), HSA_STATUS_SUCCESS);

  unsigned Iterated = 0;
  ASSERT_EQ(
      Core.hsa_executable_iterate_symbols_fn(Parent, countSymbol, &Iterated),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(Iterated, 2u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);

  Iterated = 0;
  ASSERT_EQ(
      Core.hsa_executable_iterate_symbols_fn(Parent, countSymbol, &Iterated),
      HSA_STATUS_SUCCESS);
  EXPECT_EQ(Iterated, 2u);
  EXPECT_EQ(ExecutableCreateCalls, 0u);
  EXPECT_EQ(TranslatedLoadCalls, 0u);

  hsa_executable_symbol_t First{};
  hsa_executable_symbol_t Second{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(Parent, "first_kernel",
                                                      &GpuAgent, &First),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(Parent, "second_kernel",
                                                      &GpuAgent, &Second),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(First.handle, Second.handle);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  uint64_t FirstKernelObject = 0;
  uint64_t SecondKernelObject = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          First, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &FirstKernelObject),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_symbol_get_info_fn(
                Second, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                &SecondKernelObject),
            HSA_STATUS_SUCCESS);
  uint32_t SecondKernargSize = 0;
  ASSERT_EQ(Core.hsa_executable_symbol_get_info_fn(
                Second, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &SecondKernargSize),
            HSA_STATUS_SUCCESS);
  EXPECT_EQ(SecondKernargSize, 280u);
  EXPECT_NE(FirstKernelObject, SecondKernelObject);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  TranslatedKernargSize = 280;
  createProtectedQueue();
  const auto Dispatch = [&](uint64_t KernelObject) {
    hsa_kernel_dispatch_packet_t Packet{};
    Packet.header = HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    Packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    Packet.workgroup_size_x = 1;
    Packet.workgroup_size_y = 1;
    Packet.workgroup_size_z = 1;
    Packet.grid_size_x = 1;
    Packet.grid_size_y = 1;
    Packet.grid_size_z = 1;
    Packet.kernel_object = KernelObject;
    RegisteredInterceptor(&Packet, 1, 0, RegisteredInterceptorData,
                          fakePacketWriter);
  };
  Dispatch(FirstKernelObject);
  EXPECT_EQ(ExecutableCreateCalls, 1u);
  EXPECT_EQ(TranslatedLoadCalls, 1u);
  Dispatch(FirstKernelObject);
  EXPECT_EQ(ExecutableCreateCalls, 1u);
  EXPECT_EQ(TranslatedLoadCalls, 1u);
  Dispatch(SecondKernelObject);
  EXPECT_EQ(ExecutableCreateCalls, 2u);
  EXPECT_EQ(TranslatedLoadCalls, 2u);

  EXPECT_EQ(Core.hsa_executable_destroy_fn(Parent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 3u);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ReaderDestroyCalls, 3u);
}

TEST_F(HotswapHsaToolApiTest,
       KeepsKernelIdentitiesSeparateAcrossExecutablesAndHandleReuse) {
  ASSERT_EQ(setenv("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO", "1", 1), 0);
  EnableTranslatedLoader = true;
  activate("gfx950");

  const int ObjectFile = open(COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT, O_RDONLY);
  ASSERT_NE(ObjectFile, -1);
  const COMGR::hotswap::hsa_tool::Bytes Object =
      COMGR::hotswap::hsa_tool::readWholeFile(ObjectFile);
  ASSERT_TRUE(Object);
  ASSERT_EQ(close(ObjectFile), 0);
  hsa_code_object_reader_t Reader{};
  ASSERT_EQ(Core.hsa_code_object_reader_create_from_memory_fn(
                Object->data(), Object->size(), &Reader),
            HSA_STATUS_SUCCESS);

  const hsa_executable_t FirstParent{77};
  const hsa_executable_t SecondParent{78};
  for (hsa_executable_t Parent : {FirstParent, SecondParent}) {
    ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                  Parent, GpuAgent, Reader, nullptr, nullptr),
              HSA_STATUS_SUCCESS);
    ASSERT_EQ(Core.hsa_executable_freeze_fn(Parent, nullptr),
              HSA_STATUS_SUCCESS);
  }

  hsa_executable_symbol_t FirstSymbol{};
  hsa_executable_symbol_t SecondSymbol{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(FirstParent, "vecadd",
                                                      &GpuAgent, &FirstSymbol),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(SecondParent, "vecadd",
                                                      &GpuAgent, &SecondSymbol),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(FirstSymbol.handle, SecondSymbol.handle);
  EXPECT_EQ(ExecutableCreateCalls, 0u);

  uint64_t FirstToken = 0;
  uint64_t SecondToken = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &FirstToken),
      HSA_STATUS_SUCCESS);
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          SecondSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &SecondToken),
      HSA_STATUS_SUCCESS);
  EXPECT_NE(FirstToken, SecondToken);

  EXPECT_EQ(Core.hsa_executable_destroy_fn(FirstParent), HSA_STATUS_SUCCESS);
  uint64_t StaleToken = 0;
  EXPECT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          FirstSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &StaleToken),
      HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL);
  const hsa_executable_t ReusedParent{77};
  ASSERT_EQ(Core.hsa_executable_load_agent_code_object_fn(
                ReusedParent, GpuAgent, Reader, nullptr, nullptr),
            HSA_STATUS_SUCCESS);
  ASSERT_EQ(Core.hsa_executable_freeze_fn(ReusedParent, nullptr),
            HSA_STATUS_SUCCESS);
  hsa_executable_symbol_t ReusedSymbol{};
  ASSERT_EQ(Core.hsa_executable_get_symbol_by_name_fn(ReusedParent, "vecadd",
                                                      &GpuAgent, &ReusedSymbol),
            HSA_STATUS_SUCCESS);
  EXPECT_NE(ReusedSymbol.handle, FirstSymbol.handle);
  uint64_t ReusedToken = 0;
  ASSERT_EQ(
      Core.hsa_executable_symbol_get_info_fn(
          ReusedSymbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &ReusedToken),
      HSA_STATUS_SUCCESS);
  EXPECT_NE(ReusedToken, FirstToken);

  EXPECT_EQ(Core.hsa_executable_destroy_fn(SecondParent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(Core.hsa_executable_destroy_fn(ReusedParent), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ExecutableDestroyCalls, 3u);
  EXPECT_EQ(Core.hsa_code_object_reader_destroy_fn(Reader), HSA_STATUS_SUCCESS);
  EXPECT_EQ(ReaderDestroyCalls, 1u);
}

} // namespace
