cmake_minimum_required(VERSION 3.20)
project(hotswap-interposer LANGUAGES CXX)

# Standalone, dependency-light by design. The library is one component registered
# two ways: LD_PRELOAD (KFD/sysfs identity spoof) and HSA_TOOLS_LIB (code-object
# capture + COMGR transpile). It consumes the prebuilt COMGR hotswap C API rather
# than building an LLVM tree.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# External dependencies must be supplied by the caller; there are no defaults so
# the build is portable and never silently binds a developer-specific path.
#
#   HOTSWAP_HSA_INCLUDE_DIR  - directory with hsa.h / hsa_api_trace.h. The
#       runtime's hsa_api_trace.h includes both "hsa.h" and "inc/hsa_ext_image.h",
#       so this must be the rocr-runtime source layout (an `inc/` directory whose
#       parent is also added below), not the flattened install tree.
#   HOTSWAP_COMGR_INCLUDE_DIR - directory with amd_comgr.h exposing the hotswap API.
#   HOTSWAP_COMGR_LIBRARY      - libamd_comgr.so exporting amd_comgr_hotswap_transpile.
set(HOTSWAP_HSA_INCLUDE_DIR "" CACHE PATH
    "Directory containing hsa.h / hsa_api_trace.h (rocr-runtime inc/ layout)")
set(HOTSWAP_COMGR_INCLUDE_DIR "" CACHE PATH
    "Directory containing amd_comgr.h with the hotswap API")
set(HOTSWAP_COMGR_LIBRARY "" CACHE FILEPATH
    "libamd_comgr.so exporting amd_comgr_hotswap_transpile")

foreach(_req HOTSWAP_HSA_INCLUDE_DIR HOTSWAP_COMGR_INCLUDE_DIR HOTSWAP_COMGR_LIBRARY)
  if(NOT ${_req})
    message(FATAL_ERROR
      "${_req} must be set. Point HOTSWAP_HSA_INCLUDE_DIR at a rocr-runtime "
      "inc/ directory, and HOTSWAP_COMGR_INCLUDE_DIR / HOTSWAP_COMGR_LIBRARY at "
      "a COMGR build that exports the hotswap transpile API.")
  endif()
endforeach()

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
