// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -fclangir -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm %s -o - | FileCheck %s
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -fclangir -emit-llvm %s -o %t.fallback
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -emit-llvm %s -o - >> %t.fallback
// RUN: FileCheck %s --check-prefix=FALLBACK --input-file=%t.fallback
// FALLBACK: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX:.+]](
// FALLBACK: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX]](

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -fgpu-rdc \
// RUN:   -fclangir -emit-llvm /dev/null -o - 2>&1 | FileCheck %s --check-prefix=CUDA
// CUDA: error: CUDA RDC is not implemented

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -fgpu-rdc \
// RUN:   -target-sdk-version=9.2 -fclangir -emit-llvm %s -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CUDA

#include "../CodeGenCUDA/Inputs/cuda.h"

// CHECK-DAG: @_ZL6hidden.static.b04fd23c98500190 = addrspace(1) externally_initialized global i32
static __device__ int hidden;
// CHECK-DAG: @_ZL6values.static.b04fd23c98500190 = addrspace(4) externally_initialized constant [2 x i32]
static __constant__ int values[2];
const void *host_addresses[] = {&hidden, &values};

namespace {
// CHECK: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.b04fd23c98500190(
__global__ void internal(int *p) { *p = hidden + values[0]; }
}
