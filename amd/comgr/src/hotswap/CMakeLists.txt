# Hotswap transpiler — compiled directly into amd_comgr, no separate static
# library. This subdirectory builds the optional TDM emulation-runtime
# bitcode blob and exports the hotswap source list / target-private include
# dirs / extra LLVM libs / compile definitions to the parent scope so
# `amd/comgr/CMakeLists.txt` can fold them into amd_comgr's build (and
# the test-unit / test-lit binaries can compile-in the same set).

cmake_minimum_required(VERSION 3.20)

# Resolve the LLVM source / binary tree we'll need for AMDGPU target-private
# headers and TableGen-generated `.inc` files. Three supported configurations,
# matching how the parent `amd_comgr` is built:
#
#   1. In-tree LLVM monorepo: LLVMSupport target + LLVM_MAIN_SRC_DIR are
#      already populated by the LLVM build.
#   2. Out-of-tree: parent already called find_package(LLVM); variables are
#      populated, so the find_package below is a no-op.
#   3. Standalone configure of just amd/comgr: explicit find_package call
#      resolves LLVM_DIR.
if(TARGET LLVMSupport AND DEFINED LLVM_MAIN_SRC_DIR)
  set(_hotswap_llvm_src "${LLVM_MAIN_SRC_DIR}")
else()
  if(NOT DEFINED LLVM_BUILD_MAIN_SRC_DIR OR NOT LLVM_BUILD_MAIN_SRC_DIR)
    find_package(LLVM REQUIRED CONFIG)
  endif()
  if(DEFINED LLVM_BUILD_MAIN_SRC_DIR AND LLVM_BUILD_MAIN_SRC_DIR)
    set(_hotswap_llvm_src "${LLVM_BUILD_MAIN_SRC_DIR}")
  endif()
endif()

if(NOT DEFINED _hotswap_llvm_src OR NOT _hotswap_llvm_src OR
   NOT DEFINED LLVM_BINARY_DIR OR NOT LLVM_BINARY_DIR)
  message(FATAL_ERROR
    "hotswap-transpiler requires an LLVM build tree (not an install tree). "
    "The install tree does not expose AMDGPU target-private headers "
    "(SIDefines.h, AMDGPUBaseInfo.h, AMDGPUGen*.inc). Pass "
    "-DLLVM_DIR=<llvm-build>/lib/cmake/llvm or build LLVM at "
    "<llvm-project>/build so it can be auto-detected.")
endif()

# Pull in AddLLVM.cmake so consumers (test-lit/raise_cli, the hotswap
# helper macro in test-unit/) can call `llvm_update_compile_flags` on
# their targets to mirror LLVM's RTTI/EH settings.
if(NOT COMMAND llvm_update_compile_flags)
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(AddLLVM)
endif()

# --- TDM emulation runtime helper (HIP -> bitcode -> embedded array) --------
# When `hipcc` is available we build `runtime/tdm.hip` to LLVM bitcode for
# gfx942 and embed the bytes via `cmake/EmbedBinary.cmake`. The transpiler
# parses and link-merges the bitcode into each raised IR module that uses
# the cross-target VIMAGE TENSOR helper (see `tdm_runtime.{hpp,cpp}` and
# `raiser.cpp` Phase 6.7). When hipcc is missing the embedded blob is
# empty (one-byte stub in tdm_runtime.cpp), `tdmRuntimeAvailable()`
# returns false, and the cross-target VIMAGE handler keeps today's loud
# refusal. This preserves the no-hipcc build behaviour exactly.
find_program(HIPCC hipcc)
set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP "")
if(HIPCC)
  set(_tdm_src "${CMAKE_CURRENT_SOURCE_DIR}/runtime/tdm.hip")
  set(_tdm_bc  "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime.gfx942.bc")
  set(_tdm_cpp "${CMAKE_CURRENT_BINARY_DIR}/tdm_runtime_blob.cpp")

  add_custom_command(
    OUTPUT ${_tdm_bc}
    COMMAND ${HIPCC}
            --offload-arch=gfx942
            -O2 -emit-llvm -c
            -fgpu-rdc
            --cuda-device-only
            -o ${_tdm_bc} ${_tdm_src}
    DEPENDS ${_tdm_src}
    COMMENT "Building TDM emulation runtime bitcode (gfx942)"
    VERBATIM)

  add_custom_command(
    OUTPUT ${_tdm_cpp}
    COMMAND ${CMAKE_COMMAND}
            -DINPUT=${_tdm_bc}
            -DOUTPUT=${_tdm_cpp}
            -DSYMBOL=tdm_runtime_bc
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    DEPENDS ${_tdm_bc} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake
    COMMENT "Embedding TDM runtime bitcode into C++ blob"
    VERBATIM)

  set(HOTSWAP_TDM_RUNTIME_GENERATED_CPP ${_tdm_cpp})
  set(HOTSWAP_HAVE_TDM_RUNTIME 1)
  message(STATUS "TDM runtime helper enabled (hipcc=${HIPCC})")
else()
  set(HOTSWAP_HAVE_TDM_RUNTIME 0)
  message(STATUS "TDM runtime helper DISABLED (hipcc not found); "
                 "cross-target gfx1250->gfx942 TENSOR ops will keep refusing")
endif()

# --- Hotswap source list, exported to parent scope ---------------------------
# Each commit that adds a new translation unit appends to this list. The
# parent (`amd/comgr/CMakeLists.txt`) folds these into amd_comgr's SOURCES,
# and individual test targets in `test-unit/` / `test-lit/` reuse the same
# list to compile-in just the hotswap pieces they need.
set(HOTSWAP_TRANSPILER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/raiser.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/raise_context.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reg_file.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wave_projection.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wave_size_obstruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rewrite_cross_lane_divergent.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/c5_predicate_chain_classifier.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/canonical_op_attrs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/raise_failure.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mc_state.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/opcode_map.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/canonical_op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source_hidden_args.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/setpc_analysis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_sopp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_smem.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_sopc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_sop1.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_sopk.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_sop2.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_valu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_valu_cross_lane.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_valu_small_ops.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_valu_vcmp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_valu_vop3p.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/flat_addr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mubuf_addr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_flat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_ds.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_mubuf.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_mfma.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_vopd.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/handle_vimage.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/wmma_lowering.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mxfp4_dequant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pipeline.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/translation_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/code_object_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kernarg_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/user_sgpr_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tdm_runtime.cpp
  ${HOTSWAP_TDM_RUNTIME_GENERATED_CPP}
  PARENT_SCOPE)

# AMDGPU target-private include directories. SIDefines.h, SIInstrInfo.h,
# AMDGPUBaseInfo.h are in the LLVM source tree; AMDGPUGen*.inc files are
# in the LLVM build tree. Neither set is installed by `ninja install`, so
# every consumer of the hotswap sources needs these on the include path.
set(HOTSWAP_TRANSPILER_PRIVATE_INCLUDES
  "${_hotswap_llvm_src}/lib/Target/AMDGPU"
  "${_hotswap_llvm_src}/lib/Target/AMDGPU/Utils"
  "${LLVM_BINARY_DIR}/lib/Target/AMDGPU"
  PARENT_SCOPE)

# Extra LLVM libraries the hotswap raiser pulls in beyond what comgr core
# already links (codegen, AMDGPU target backend, MIR / GlobalISel / SelDAG
# infrastructure).
set(HOTSWAP_TRANSPILER_LLVM_LIBS
  LLVMAMDGPUDisassembler
  LLVMAMDGPUCodeGen
  LLVMAMDGPUAsmParser
  LLVMAMDGPUDesc
  LLVMAMDGPUInfo
  LLVMAMDGPUUtils
  LLVMAsmPrinter
  LLVMSelectionDAG
  LLVMGlobalISel
  LLVMMIRParser
  LLVMScalarOpts
  LLVMTransformUtils
  LLVMAnalysis
  LLVMipo
  LLVMInstCombine
  LLVMAggressiveInstCombine
  LLVMInstrumentation
  LLVMVectorize
  LLVMPasses
  LLVMFrontendOffloading
  LLVMFrontendOpenMP
  LLVMLinker
  LLVMObjCARCOpts
  PARENT_SCOPE)

set(HOTSWAP_TRANSPILER_COMPILE_DEFS
  HOTSWAP_HAVE_TDM_RUNTIME=${HOTSWAP_HAVE_TDM_RUNTIME}
  LLVM_TOOLS_DIR="${LLVM_TOOLS_BINARY_DIR}"
  PARENT_SCOPE)
