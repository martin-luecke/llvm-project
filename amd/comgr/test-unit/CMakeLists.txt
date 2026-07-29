# GTest discovery for Comgr unit tests. Three sources, in priority order:
#
# 1. In-tree llvm_gtest target (LLVM monorepo build with utils/unittest/).
# 2. Vanilla GTest CMake package (find_package(GTest CONFIG)). Used by
#    superproject builds (e.g. TheRock) that supply googletest via the
#    standard GTest:: imported targets. This avoids forcing the LLVM
#    build to enable LLVM_INSTALL_GTEST just to test Comgr.
# 3. LLVM-installed gtest (-DLLVM_INSTALL_GTEST=ON on the LLVM build).
#    Legacy fallback for installs that ship llvm_gtest alongside LLVM.
#
# If none is available, skip the test-unit suite with a warning rather
# than failing configure.
if(TARGET llvm_gtest)
  set(COMGR_GTEST_LIBS llvm_gtest_main llvm_gtest)
  set(COMGR_GTEST_INCLUDE_DIRS "")
else()
  find_package(GTest CONFIG QUIET)
  if(GTest_FOUND)
    set(COMGR_GTEST_LIBS GTest::gtest_main GTest::gtest)
    set(COMGR_GTEST_INCLUDE_DIRS "")
  else()
    find_library(COMGR_LLVM_GTEST_LIB llvm_gtest
      PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_library(COMGR_LLVM_GTEST_MAIN_LIB llvm_gtest_main
      PATHS ${LLVM_LIBRARY_DIRS} NO_DEFAULT_PATH)
    find_path(COMGR_LLVM_GTEST_INCLUDE_DIR gtest/gtest.h
      PATHS ${LLVM_INCLUDE_DIRS} PATH_SUFFIXES llvm-gtest NO_DEFAULT_PATH)
    if(COMGR_LLVM_GTEST_LIB AND COMGR_LLVM_GTEST_MAIN_LIB
        AND COMGR_LLVM_GTEST_INCLUDE_DIR)
      set(COMGR_GTEST_LIBS
        ${COMGR_LLVM_GTEST_MAIN_LIB} ${COMGR_LLVM_GTEST_LIB})
      set(COMGR_GTEST_INCLUDE_DIRS ${COMGR_LLVM_GTEST_INCLUDE_DIR})
    else()
      message(WARNING
        "Comgr test-unit skipped: no llvm_gtest target, no GTest "
        "CMake package, and no LLVM-installed gtest found at "
        "${LLVM_INCLUDE_DIRS}/llvm-gtest. Provide one of: in-tree LLVM "
        "with utils/unittest/, find_package(GTest), or "
        "-DLLVM_INSTALL_GTEST=ON on the LLVM build.")
      return()
    endif()
  endif()
endif()

function(comgr_add_amdgpu_private_include_dirs target)
  if(DEFINED LLVM_BUILD_MAIN_SRC_DIR AND LLVM_BUILD_MAIN_SRC_DIR)
    set(_comgr_llvm_main_src_dir "${LLVM_BUILD_MAIN_SRC_DIR}")
  elseif(DEFINED LLVM_MAIN_SRC_DIR AND LLVM_MAIN_SRC_DIR)
    set(_comgr_llvm_main_src_dir "${LLVM_MAIN_SRC_DIR}")
  else()
    message(FATAL_ERROR
      "Comgr test target ${target} requires LLVM source-tree paths for "
      "AMDGPU target-private headers. Expected LLVM_MAIN_SRC_DIR or "
      "LLVM_BUILD_MAIN_SRC_DIR to be set by the LLVM build.")
  endif()

  # Tests that pull in transpiler internals (wave-projection.h, opcode-map.h,
  # ...) need AMDGPU target-private headers too: SIDefines.h, SIInstrInfo.h,
  # AMDGPUBaseInfo.h live in the LLVM source tree, AMDGPUGen*.inc files in the
  # LLVM build tree.
  target_include_directories(${target} PRIVATE
    "${_comgr_llvm_main_src_dir}/lib/Target/AMDGPU"
    "${_comgr_llvm_main_src_dir}/lib/Target/AMDGPU/Utils"
    "${LLVM_BINARY_DIR}/lib/Target/AMDGPU")
endfunction()

# Wire up a test target against the hotswap transpiler OBJECT lib + the
# COMGR-internal helpers it calls (`getDeviceLibraries`,
# `metadata::walkElfMetadataIntoDoc`, and `metadata::getElfIsaNameFromBuffer`).
# Test binaries don't link amd_comgr.so (its export map hides everything
# except the C entry points), so the helper definitions must be compiled
# in directly. `-ffunction-sections -fdata-sections` + `--gc-sections`
# lets the linker drop unreferenced comgr-metadata.cpp entries
# (parseTargetIdentifier, the IsaInfos-driven validation, etc.) without
# dragging in comgr.cpp's transitive dependency closure.
function(comgr_link_hotswap_transpiler target)
  # `amd_comgr` provides the public C entry points
  # (`amd_comgr_create_data` / `set_data` / `release_data`) that
  # `code-object-utils.cpp` calls into to wrap raw ELF bytes for the
  # `metadata::getMetadataRoot` / `metadata::getElfIsaName` paths.
  target_link_libraries(${target} PRIVATE hotswap::transpiler amd_comgr)
  target_sources(${target} PRIVATE
    $<TARGET_OBJECTS:comgr-device-libs-internal>
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/comgr-metadata.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/comgr-symbol.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../test-shared/standalone-init.cpp)
  target_include_directories(${target} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_BINARY_DIR}/../include)
  comgr_add_amdgpu_private_include_dirs(${target})
  if(NOT MSVC)
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
  endif()
endfunction()

# Pin C++17 and mirror LLVM's RTTI setting (avoids type_info link errors
# against LLVM libs built with RTTI off, including the hotswap transpiler
# OBJECT lib which carries its own non-RTTI / non-exceptions compile
# flags via llvm_update_compile_flags()).
function(comgr_configure_test_target target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED Yes
    CXX_EXTENSIONS No)
  if(NOT LLVM_ENABLE_RTTI)
    if(MSVC)
      target_compile_options(${target} PRIVATE /GR-)
    else()
      target_compile_options(${target} PRIVATE -fno-rtti)
    endif()
  endif()
endfunction()

# -- HotswapElfTests ----------------------------------------------------------
#
# Lightweight tests for the ELF layer in comgr-hotswap-elf.cpp. Needs only
# llvm::object for ELF parsing; no MC state constructed.

add_executable(HotswapElfTests
  HotswapElfTest.cpp
  ../src/comgr-hotswap-elf.cpp
  # COMGR::env::shouldEmitVerboseLogs() is referenced by the inline
  # COMGR::hotswap::log() helper in comgr-hotswap-internal.h; link its
  # definition so non-inlined builds (e.g. -O0 / ASan) resolve.
  ../src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_ELF_LIBS
  Object Support TargetParser)

target_link_libraries(HotswapElfTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_ELF_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapElfTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(HotswapElfTests)

# -- HotswapMCTests -----------------------------------------------------------
#
# MC-layer tests need the AMDGPU backend (TargetRegistry, instruction
# definitions, disassembler, asm parser, and code emitter) so the test
# binary can stand up a real LLVMState for gfx1250 and exercise the
# assemble / decode / encode primitives.

add_executable(HotswapMCTests
  HotswapMCTest.cpp
  ../src/comgr-hotswap-b0a0.cpp
  ../src/comgr-hotswap-elf.cpp
  ../src/comgr-hotswap-llvm.cpp
  ../src/comgr-hotswap-patch-inplace.cpp
  ../src/comgr-hotswap-patch-trampoline.cpp
  ../src/comgr-hotswap-patch-vop3px2-src2.cpp
  ../src/comgr-hotswap-patch-wmma-hazard.cpp
  ../src/comgr-hotswap-patch-wmma-split.cpp
  ../src/comgr-env.cpp)

llvm_map_components_to_libnames(COMGR_TEST_UNIT_MC_LIBS
  MC
  MCDisassembler
  MCParser
  Object
  Support
  TargetParser
  AMDGPUAsmParser
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUDisassembler
  AMDGPUInfo)

target_link_libraries(HotswapMCTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${COMGR_TEST_UNIT_MC_LIBS}
  ${LLVM_PTHREAD_LIB})

target_include_directories(HotswapMCTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(HotswapMCTests)

# -- RaiserScaffoldingTests ---------------------------------------------------
#
# Pins the bare-bones raiser scaffolding contract (`hotswap/raiser.hpp`):
# empty input produces a verifyModule-clean `llvm::Module` with a single
# `AMDGPU_KERNEL` `ret void` function, plus the failure paths for missing
# kernel descriptor / malformed ISA.

add_executable(RaiserScaffoldingTests
  RaiserScaffoldingTest.cpp)

target_link_libraries(RaiserScaffoldingTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(RaiserScaffoldingTests)

target_include_directories(RaiserScaffoldingTests PRIVATE
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  ${COMGR_GTEST_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../include>)

comgr_configure_test_target(RaiserScaffoldingTests)

# -- CodeObjectUtilsTests ---------------------------------------------------------------

add_executable(CodeObjectUtilsTests
  CodeObjectUtilsTest.cpp)

target_link_libraries(CodeObjectUtilsTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(CodeObjectUtilsTests)

target_include_directories(CodeObjectUtilsTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(CodeObjectUtilsTests)

# -- MCContextInlineSrcmgrTests -----------------------------------------------------------------

add_executable(MCContextInlineSrcmgrTests
  MCContextInlineSrcmgrTest.cpp)

target_link_libraries(MCContextInlineSrcmgrTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(MCContextInlineSrcmgrTests)

target_include_directories(MCContextInlineSrcmgrTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(MCContextInlineSrcmgrTests)

# -- CanonicalOpAttrsTests ----------------------------------------------------------

add_executable(CanonicalOpAttrsTests
  CanonicalOpAttrsTest.cpp)

target_link_libraries(CanonicalOpAttrsTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(CanonicalOpAttrsTests)

target_include_directories(CanonicalOpAttrsTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(CanonicalOpAttrsTests)

# -- OpcodeMapTests -----------------------------------------------------------------

add_executable(OpcodeMapTests
  OpcodeMapTest.cpp)

target_link_libraries(OpcodeMapTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(OpcodeMapTests)

target_include_directories(OpcodeMapTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(OpcodeMapTests)

# -- KernargLayoutTests -----------------------------------------------------------------

add_executable(KernargLayoutTests
  KernargLayoutTest.cpp)

target_link_libraries(KernargLayoutTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(KernargLayoutTests)

target_include_directories(KernargLayoutTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(KernargLayoutTests)

# -- WaveProjectionTests -----------------------------------------------------------------

add_executable(WaveProjectionTests
  WaveProjectionTest.cpp)

target_link_libraries(WaveProjectionTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(WaveProjectionTests)

target_include_directories(WaveProjectionTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(WaveProjectionTests)

# -- SourceHiddenArgsTests -----------------------------------------------------------------

add_executable(SourceHiddenArgsTests
  SourceHiddenArgsTest.cpp)

target_link_libraries(SourceHiddenArgsTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(SourceHiddenArgsTests)

target_include_directories(SourceHiddenArgsTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(SourceHiddenArgsTests)

# -- MXFP4DequantTests -----------------------------------------------------------------

add_executable(MXFP4DequantTests
  MXFP4DequantTest.cpp)

target_link_libraries(MXFP4DequantTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(MXFP4DequantTests)

target_include_directories(MXFP4DequantTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(MXFP4DequantTests)

# -- C5PredicateChainTests -----------------------------------------------------------------

add_executable(C5PredicateChainTests
  C5PredicateChainTest.cpp)

target_link_libraries(C5PredicateChainTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(C5PredicateChainTests)

target_include_directories(C5PredicateChainTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(C5PredicateChainTests)

# -- TranslationCacheTests -----------------------------------------------------------------

add_executable(TranslationCacheTests
  TranslationCacheTest.cpp)

target_link_libraries(TranslationCacheTests PRIVATE
  ${COMGR_GTEST_LIBS}
  ${LLVM_PTHREAD_LIB})
comgr_link_hotswap_transpiler(TranslationCacheTests)

target_include_directories(TranslationCacheTests PRIVATE
  ${COMGR_GTEST_INCLUDE_DIRS})

comgr_configure_test_target(TranslationCacheTests)

# -- HotswapHsaToolDispatchTests ---------------------------------------------

if(COMGR_BUILD_HOTSWAP_HSA_TOOL)
  set(_hotswap_hsa_tool_global_object
    ${CMAKE_CURRENT_BINARY_DIR}/hotswap-hsa-tool-shared-global.co)
  set(_hotswap_hsa_tool_two_kernel_object
    ${CMAKE_CURRENT_BINARY_DIR}/hotswap-hsa-tool-two-kernel.co)
  set(_hotswap_hsa_tool_stripped_object
    ${CMAKE_CURRENT_BINARY_DIR}/vecadd-gfx950-stripped.co)
  add_custom_command(
    OUTPUT ${_hotswap_hsa_tool_global_object}
    COMMAND $<TARGET_FILE:llvm-mc> -triple=amdgcn-amd-amdhsa -mcpu=gfx950
      ${CMAKE_CURRENT_SOURCE_DIR}/hotswap-hsa-tool-shared-global.s
      -filetype=obj -o ${_hotswap_hsa_tool_global_object}.o
    COMMAND $<TARGET_FILE:lld> -flavor gnu -shared --strip-all
      ${_hotswap_hsa_tool_global_object}.o
      -o ${_hotswap_hsa_tool_global_object}
    DEPENDS hotswap-hsa-tool-shared-global.s llvm-mc lld
    VERBATIM)
  add_custom_target(HotswapHsaToolTestObjects
    DEPENDS ${_hotswap_hsa_tool_global_object}
      ${_hotswap_hsa_tool_two_kernel_object}
      ${_hotswap_hsa_tool_stripped_object})
  add_custom_command(
    OUTPUT ${_hotswap_hsa_tool_two_kernel_object}
    COMMAND $<TARGET_FILE:llvm-mc> -triple=amdgcn-amd-amdhsa -mcpu=gfx950
      ${CMAKE_CURRENT_SOURCE_DIR}/hotswap-hsa-tool-two-kernel.s
      -filetype=obj -o ${_hotswap_hsa_tool_two_kernel_object}.o
    COMMAND $<TARGET_FILE:lld> -flavor gnu -shared
      ${_hotswap_hsa_tool_two_kernel_object}.o
      -o ${_hotswap_hsa_tool_two_kernel_object}
    DEPENDS hotswap-hsa-tool-two-kernel.s llvm-mc lld
    VERBATIM)
  add_custom_command(
    OUTPUT ${_hotswap_hsa_tool_stripped_object}
    COMMAND $<TARGET_FILE:llvm-objcopy> --strip-all
      ${CMAKE_CURRENT_SOURCE_DIR}/../test-lit/vecadd_gfx950.co
      ${_hotswap_hsa_tool_stripped_object}
    DEPENDS ../test-lit/vecadd_gfx950.co llvm-objcopy
    VERBATIM)
  add_executable(HotswapHsaToolDispatchTests
    HotswapHsaToolApiTest.cpp
    HotswapHsaToolDispatchTest.cpp
    ../tools/hotswap-hsa-tool/hotswap-dispatch.cpp
    ../tools/hotswap-hsa-tool/hotswap-object.cpp
    ../tools/hotswap-hsa-tool/hotswap-platform-io-posix.cpp
    ../tools/hotswap-hsa-tool/hotswap-proof-posix.cpp)
  target_link_libraries(HotswapHsaToolDispatchTests PRIVATE
    comgr-hotswap-hsa-tool
    ${COMGR_GTEST_LIBS}
    ${LLVM_PTHREAD_LIB}
    LLVMObject
    LLVMSupport)
  target_include_directories(HotswapHsaToolDispatchTests PRIVATE
    ${COMGR_GTEST_INCLUDE_DIRS}
    ../tools/hotswap-hsa-tool
    ${COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR}
    ${COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR}/inc)
  target_compile_definitions(HotswapHsaToolDispatchTests PRIVATE
    "COMGR_HOTSWAP_HSA_TOOL_TEST_OBJECT=\"${CMAKE_CURRENT_SOURCE_DIR}/../test-lit/vecadd_gfx950.co\""
    "COMGR_HOTSWAP_HSA_TOOL_GLOBAL_TEST_OBJECT=\"${_hotswap_hsa_tool_global_object}\""
    "COMGR_HOTSWAP_HSA_TOOL_TWO_KERNEL_TEST_OBJECT=\"${_hotswap_hsa_tool_two_kernel_object}\""
    "COMGR_HOTSWAP_HSA_TOOL_STRIPPED_TEST_OBJECT=\"${_hotswap_hsa_tool_stripped_object}\"")
  add_dependencies(HotswapHsaToolDispatchTests HotswapHsaToolTestObjects)
  comgr_configure_test_target(HotswapHsaToolDispatchTests)
endif()

# Register every test binary with the test-unit / check-comgr plumbing.
add_custom_target(test-unit
  COMMAND $<TARGET_FILE:HotswapElfTests>
  COMMAND $<TARGET_FILE:HotswapMCTests>
  COMMAND $<TARGET_FILE:CodeObjectUtilsTests>
  COMMAND $<TARGET_FILE:MCContextInlineSrcmgrTests>
  COMMAND $<TARGET_FILE:OpcodeMapTests>
  COMMAND $<TARGET_FILE:CanonicalOpAttrsTests>
  COMMAND $<TARGET_FILE:RaiserScaffoldingTests>
  COMMAND $<TARGET_FILE:KernargLayoutTests>
  COMMAND $<TARGET_FILE:WaveProjectionTests>
  COMMAND $<TARGET_FILE:SourceHiddenArgsTests>
  COMMAND $<TARGET_FILE:MXFP4DequantTests>
  COMMAND $<TARGET_FILE:C5PredicateChainTests>
  COMMAND $<TARGET_FILE:TranslationCacheTests>)
add_dependencies(test-unit HotswapElfTests HotswapMCTests
  RaiserScaffoldingTests
  CodeObjectUtilsTests
  MCContextInlineSrcmgrTests
  OpcodeMapTests
  CanonicalOpAttrsTests
  KernargLayoutTests
  WaveProjectionTests
  SourceHiddenArgsTests
  MXFP4DequantTests
  C5PredicateChainTests
  TranslationCacheTests)
if(COMGR_BUILD_HOTSWAP_HSA_TOOL)
  add_dependencies(test-unit HotswapHsaToolDispatchTests)
  add_custom_command(TARGET test-unit POST_BUILD
    COMMAND $<TARGET_FILE:HotswapHsaToolDispatchTests>)
endif()
add_dependencies(check-comgr test-unit)
