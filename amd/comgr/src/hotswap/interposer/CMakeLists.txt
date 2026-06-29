cmake_minimum_required(VERSION 3.20)
project(hotswap-interposer LANGUAGES CXX)

# Standalone, dependency-light by design. The library is one component registered
# two ways: LD_PRELOAD (KFD/sysfs identity spoof) and HSA_TOOLS_LIB (code-object
# capture + COMGR transpile). It consumes the prebuilt COMGR hotswap C API rather
# than building an LLVM tree.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# HSA headers (hsa.h, hsa_api_trace.h) for the tool half. The runtime's
# hsa_api_trace.h includes both "hsa.h" and "inc/hsa_ext_image.h", so it needs
# the rocr-runtime source layout (the inc/ dir and its parent) rather than the
# flattened install tree. Defaults to the in-tree rocr-runtime that the matching
# gfx1250-aware ROCr is built from.
set(HOTSWAP_HSA_INCLUDE_DIR
    "$ENV{HOME}/development/rocm-systems/projects/rocr-runtime/runtime/hsa-runtime/inc"
    CACHE PATH "Directory containing hsa.h / hsa_api_trace.h")

# Prebuilt COMGR with the hotswap transpile API + its matching public header.
set(HOTSWAP_COMGR_INCLUDE_DIR
    "$ENV{HOME}/development/llvm-lds-spill/.worktrees/qwen-moe/comgr-build/include"
    CACHE PATH "Directory containing amd_comgr.h with the hotswap API")
set(HOTSWAP_COMGR_LIBRARY
    "$ENV{HOME}/development/llvm-lds-spill/.worktrees/qwen-moe/comgr-build/libamd_comgr.so"
    CACHE FILEPATH "Prebuilt libamd_comgr.so exporting amd_comgr_hotswap_transpile")

add_library(hotswap-interposer SHARED
  src/interposer.cpp
  src/topology_spoof.cpp
  src/transpile.cpp
  src/hsa_tool.cpp)

target_include_directories(hotswap-interposer PRIVATE
  src
  "${HOTSWAP_HSA_INCLUDE_DIR}"
  "${HOTSWAP_HSA_INCLUDE_DIR}/.."
  "${HOTSWAP_COMGR_INCLUDE_DIR}")
# Default visibility: the extern "C" libc hooks must be exported for LD_PRELOAD
# interposition, as must OnLoad/OnUnload for HSA tool discovery.
set_target_properties(hotswap-interposer PROPERTIES
  CXX_VISIBILITY_PRESET default
  OUTPUT_NAME hotswap-interposer)
target_compile_options(hotswap-interposer PRIVATE -Wall -Wextra)
target_link_libraries(hotswap-interposer PRIVATE
  "${HOTSWAP_COMGR_LIBRARY}"
  ${CMAKE_DL_LIBS})

# Ensure the hotswap-capable COMGR is found at runtime ahead of any system one.
get_filename_component(_hotswap_comgr_dir "${HOTSWAP_COMGR_LIBRARY}" DIRECTORY)
set_target_properties(hotswap-interposer PROPERTIES
  BUILD_RPATH "${_hotswap_comgr_dir}")
