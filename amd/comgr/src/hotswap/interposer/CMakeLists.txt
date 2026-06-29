cmake_minimum_required(VERSION 3.20)
project(hotswap-interposer LANGUAGES CXX)

# Standalone, dependency-light by design: the KFD/sysfs identity spoof (Phase 1)
# needs only libc + libdl. The code-object capture/transpile half (Phase 2) links
# the prebuilt COMGR hotswap C API and is added behind an option once present, so
# this target never pulls in an LLVM build tree.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(hotswap-interposer SHARED
  src/interposer.cpp
  src/topology_spoof.cpp)

target_include_directories(hotswap-interposer PRIVATE src)
set_target_properties(hotswap-interposer PROPERTIES
  CXX_VISIBILITY_PRESET default
  OUTPUT_NAME hotswap-interposer)
target_compile_options(hotswap-interposer PRIVATE -Wall -Wextra)
target_link_libraries(hotswap-interposer PRIVATE ${CMAKE_DL_LIBS})
