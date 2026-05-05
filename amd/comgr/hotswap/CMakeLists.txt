cmake_minimum_required(VERSION 3.20)
project(hotswap-transpiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- LLVM ---------------------------------------------------------------------
if(NOT TARGET LLVMSupport)
  find_package(LLVM REQUIRED CONFIG)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# --- Library ------------------------------------------------------------------
add_library(hotswap-transpiler STATIC
  raiser.cpp
  raise_failure.cpp
)

if(NOT TARGET hotswap::transpiler)
  add_library(hotswap::transpiler ALIAS hotswap-transpiler)
endif()

# Match LLVM's compile flags (no-rtti, no-exceptions) — every LLVM type we
# consume relies on LLVM's hand-rolled RTTI rather than C++ typeid; mixing
# RTTI-on TUs with RTTI-off LLVM libs produces undefined-reference errors
# for `typeinfo` symbols on any class with a virtual method.
llvm_update_compile_flags(hotswap-transpiler)

set_target_properties(hotswap-transpiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Public include root so consumers can `#include "hotswap/raiser.hpp"`.
target_include_directories(hotswap-transpiler PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)

target_link_libraries(hotswap-transpiler
  PUBLIC
    LLVMCore
    LLVMSupport
    LLVMTargetParser
)
