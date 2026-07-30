# Unit tests for the hotswap raiser (ISA-to-IR transpiler scaffolding).

# -- RaiserScaffoldingTests ---------------------------------------------------
#
# Pins the bare-bones raiser scaffolding contract (`hotswap/raiser/raiser.h`):
# empty input produces a verifyModule-clean `llvm::Module` with a single
# `AMDGPU_KERNEL` `ret void` function, plus the failure paths for missing
# kernel descriptor / malformed ISA.

add_executable(RaiserScaffoldingTests
  RaiserScaffoldingTest.cpp)

target_link_libraries(RaiserScaffoldingTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  ${LLVM_PTHREAD_LIB})

target_include_directories(RaiserScaffoldingTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

comgr_configure_test_target(RaiserScaffoldingTests)

# -- KernargAbiLayoutTests ----------------------------------------------------
#
# Covers the source kernel-argument ABI reconstruction: the user-SGPR layout
# derived from the kernel descriptor, hidden-argument byte classification, and
# the IR synthesis of source hidden argument values.

add_executable(KernargAbiLayoutTests
  KernargAbiLayoutTest.cpp)

target_link_libraries(KernargAbiLayoutTests PRIVATE
  ${COMGR_GTEST_LIBS}
  hotswap::raiser
  hotswap::decoder
  hotswap::common
  ${LLVM_PTHREAD_LIB})

# UserSgprLayout's header pulls in the ISA profile, which includes AMDGPU
# target-private headers (AMDGPUMCTargetDesc.h, AMDGPUBaseInfo.h) and their
# TableGen-generated .inc files. Resolve the LLVM AMDGPU source tree the same
# way hotswap-decoder does; the LLVM libraries themselves come transitively
# from hotswap::raiser, so its dylib/static linkage choice is honored.
set(HOTSWAP_AMDGPU_SRC_DIR "")
foreach(HotswapLLVMRoot
    "${LLVM_MAIN_SRC_DIR}"
    "${LLVM_BUILD_MAIN_SRC_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../llvm")
  if(HotswapLLVMRoot AND
     EXISTS "${HotswapLLVMRoot}/lib/Target/AMDGPU/SIDefines.h")
    get_filename_component(HOTSWAP_AMDGPU_SRC_DIR
      "${HotswapLLVMRoot}/lib/Target/AMDGPU" ABSOLUTE)
    break()
  endif()
endforeach()
if(NOT HOTSWAP_AMDGPU_SRC_DIR)
  message(FATAL_ERROR
    "KernargAbiLayoutTests: cannot find the LLVM AMDGPU target sources "
    "(SIDefines.h); set LLVM_MAIN_SRC_DIR to the llvm source directory.")
endif()

target_include_directories(KernargAbiLayoutTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
  "${HOTSWAP_AMDGPU_SRC_DIR}"
  "${HOTSWAP_AMDGPU_SRC_DIR}/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU")

comgr_configure_test_target(KernargAbiLayoutTests)
