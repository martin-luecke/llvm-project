// REQUIRES: x86-registered-target, amdgpu-registered-target, nvptx-registered-target

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HIP-FIXED
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=HIP-FIXED

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=CUDA-FIXED
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=CUDA-FIXED

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -fclangir -emit-llvm %s -o %t.hip
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -emit-llvm %s -o - >> %t.hip
// RUN: FileCheck %s --check-prefix=HIP --input-file=%t.hip

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=2 -fclangir -emit-llvm %s -o %t.hip
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=2 -emit-llvm %s -o - >> %t.hip
// RUN: FileCheck %s --check-prefix=HIP --input-file=%t.hip

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -fclangir -emit-llvm %s -o %t.hip
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -emit-llvm %s -o - >> %t.hip
// RUN: FileCheck %s --check-prefix=HIP --input-file=%t.hip

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -fclangir -emit-llvm %s -o %t.cuda
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x cuda -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -emit-llvm %s -o - >> %t.cuda
// RUN: FileCheck %s --check-prefix=CUDA --input-file=%t.cuda

// RUN: echo 'namespace { __attribute__((global)) void kernel() {} }' \
// RUN:   | not %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:     -fgpu-rdc -fclangir -emit-llvm -o - - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DIAG
// DIAG: error: cannot open file

#include "../CodeGenCUDA/Inputs/cuda.h"
#line 1 "missing-device-source.hip"

// HIP-FIXED-DAG: @_ZL6hidden.static.b04fd23c98500190 = addrspace(1) externally_initialized global i32 0
// CUDA-FIXED-DAG: @_ZL6hidden__static__b04fd23c98500190 = addrspace(1) externally_initialized global i32 0
// HIP-DAG: @_ZL6hidden.static.[[SUFFIX:.+]] = addrspace(1) externally_initialized global i32 0
// HIP-DAG: @_ZL6hidden.static.[[SUFFIX]] = addrspace(1) externally_initialized global i32 0
// CUDA-DAG: @_ZL6hidden__static__[[SUFFIX:.+]] = addrspace(1) externally_initialized global i32 0
// CUDA-DAG: @_ZL6hidden__static__[[SUFFIX]] = addrspace(1) externally_initialized global i32 0
static __device__ int hidden;

// HIP-FIXED-DAG: @_ZL6values.static.b04fd23c98500190 = addrspace(4) externally_initialized constant [2 x i32] zeroinitializer
// CUDA-FIXED-DAG: @_ZL6values__static__b04fd23c98500190 = addrspace(4) externally_initialized constant [2 x i32] zeroinitializer
// HIP-DAG: @_ZL6values.static.[[SUFFIX]] = addrspace(4) externally_initialized constant [2 x i32] zeroinitializer
// HIP-DAG: @_ZL6values.static.[[SUFFIX]] = addrspace(4) externally_initialized constant [2 x i32] zeroinitializer
// CUDA-DAG: @_ZL6values__static__[[SUFFIX]] = addrspace(4) externally_initialized constant [2 x i32] zeroinitializer
// CUDA-DAG: @_ZL6values__static__[[SUFFIX]] = addrspace(4) externally_initialized constant [2 x i32] zeroinitializer
static __constant__ int values[2];
const void *host_addresses[] = {&hidden, &values};

namespace {
// HIP-FIXED: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.b04fd23c98500190(
// CUDA-FIXED: define {{.+}}ptx_kernel void @_ZN12_GLOBAL__N_18internalEPi__intern__b04fd23c98500190(
// HIP-DAG: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX]](
// HIP-DAG: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX]](
// CUDA-DAG: define {{.+}}ptx_kernel void @_ZN12_GLOBAL__N_18internalEPi__intern__[[SUFFIX]](
// CUDA-DAG: define {{.+}}ptx_kernel void @_ZN12_GLOBAL__N_18internalEPi__intern__[[SUFFIX]](
__global__ void internal(int *p) { *p = hidden + values[0]; }
}
