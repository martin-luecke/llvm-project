// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:   -fgpu-rdc -cuid=abc -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --implicit-check-not=__hip_module_ctor \
// RUN:   --implicit-check-not=.offloading.entry._Z11declarationPi
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --implicit-check-not=__hip_module_ctor \
// RUN:   --implicit-check-not=.offloading.entry._Z11declarationPi
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE
// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -fclangir -emit-llvm %s -o %t.fallback
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -emit-llvm %s -o - >> %t.fallback
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -fclangir -emit-llvm %s -o - >> %t.fallback
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -DUSER_MACRO=1 -emit-llvm %s -o - >> %t.fallback
// RUN: FileCheck %s --check-prefix=FALLBACK --input-file=%t.fallback
// FALLBACK: c"_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX:.+]]\00"
// FALLBACK: c"_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX]]\00"
// FALLBACK: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX]](
// FALLBACK: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.[[SUFFIX]](

#include "../CodeGenCUDA/Inputs/cuda.h"

// CHECK-DAG: @_Z6kernelPi = constant ptr @_Z21__device_stub__kernelPi
// CHECK-DAG: @[[NAME:.+]] = {{.+}}c"_Z6kernelPi\00", section ".llvm.rodata.offloading"
// CHECK-DAG: @.offloading.entry._Z6kernelPi = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 0, ptr @_Z6kernelPi, ptr @[[NAME]], i64 0, i64 0, ptr null }, section "llvm_offload_entries"
// DEVICE: define {{.+}}amdgpu_kernel void @_Z6kernelPi
__global__ void kernel(int *p) { *p = 1; }

namespace {
// CHECK-DAG: @_ZN12_GLOBAL__N_18internalEPi = internal constant ptr @_ZN12_GLOBAL__N_123__device_stub__internalEPi
// CHECK-DAG: @[[INTERNAL_NAME:.+]] = {{.+}}c"_ZN12_GLOBAL__N_18internalEPi.intern.b04fd23c98500190\00", section ".llvm.rodata.offloading"
// CHECK-DAG: @.offloading.entry._ZN12_GLOBAL__N_18internalEPi.intern.b04fd23c98500190 = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 0, ptr @_ZN12_GLOBAL__N_18internalEPi, ptr @[[INTERNAL_NAME]], i64 0, i64 0, ptr null }, section "llvm_offload_entries"
// DEVICE: define {{.+}}amdgpu_kernel void @_ZN12_GLOBAL__N_18internalEPi.intern.b04fd23c98500190
__global__ void internal(int *p) { *p = 2; }
}

__global__ void declaration(int *);

void launch(int *p) {
  kernel<<<1, 1>>>(p);
  internal<<<1, 1>>>(p);
  declaration<<<1, 1>>>(p);
}
