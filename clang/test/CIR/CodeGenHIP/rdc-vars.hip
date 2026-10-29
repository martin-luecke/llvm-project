// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -cuid=abc -fclangir -emit-llvm %s -o - | FileCheck %s \
// RUN:   --implicit-check-not=__hip_module_ctor \
// RUN:   --implicit-check-not=.offloading.entry.unused_extern \
// RUN:   --implicit-check-not=.offloading.entry.used_extern \
// RUN:   --implicit-check-not=.offloading.entry.extern_surf \
// RUN:   --implicit-check-not=.offloading.entry.extern_tex
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -cuid=abc -emit-llvm %s -o - | FileCheck %s \
// RUN:   --implicit-check-not=__hip_module_ctor \
// RUN:   --implicit-check-not=.offloading.entry.unused_extern \
// RUN:   --implicit-check-not=.offloading.entry.used_extern \
// RUN:   --implicit-check-not=.offloading.entry.extern_surf \
// RUN:   --implicit-check-not=.offloading.entry.extern_tex
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -cuid=abc -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir | FileCheck %s --check-prefix=CIR
// RUN: cir-opt %t.cir --cir-to-llvm | cir-translate --mlir-to-llvmir \
// RUN:   | FileCheck %s --implicit-check-not=__hip_module_ctor
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE
// RUN: %clang_cc1 -triple amdgpu-amd-amdhsa -x hip -fcuda-is-device \
// RUN:   -fgpu-rdc -cuid=abc -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE
// REQUIRES: amdgpu-registered-target

#include "../CodeGenCUDA/Inputs/cuda.h"

// CHECK-DAG: @.offloading.entry.var = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 0, ptr @var, ptr @{{.+}}, i64 4, i64 0, ptr null }, section "llvm_offload_entries"
__device__ int var;
// CHECK-DAG: @.offloading.entry.constant = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 16, ptr @constant, ptr @{{.+}}, i64 8, i64 0, ptr null }, section "llvm_offload_entries"
__constant__ int constant[2];
extern __device__ int used_extern;
extern __device__ int unused_extern;

// CHECK-DAG: @[[STATIC_NAME:.+]] = {{.+}}c"_ZL6hidden.static.b04fd23c98500190\00"
// CHECK-DAG: @.offloading.entry._ZL6hidden.static.b04fd23c98500190 = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 0, ptr @_ZL6hidden, ptr @[[STATIC_NAME]], i64 4, i64 0, ptr null }, section "llvm_offload_entries"
// DEVICE-DAG: @_ZL6hidden.static.b04fd23c98500190 = addrspace(1) externally_initialized global i32
static __device__ int hidden;

// CHECK-DAG: @.offloading.entry.managed = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 1, ptr @managed.managed, ptr @{{.+}}, i64 8, i64 16, ptr @managed }, section "llvm_offload_entries"
__managed__ __attribute__((aligned(16))) int managed[2];
// CHECK-DAG: @.offloading.entry.extern_managed = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 9, ptr @extern_managed.managed, ptr @{{.+}}, i64 4, i64 4, ptr @extern_managed }, section "llvm_offload_entries"
extern __managed__ int extern_managed;
// CHECK-DAG: @.offloading.entry._ZL14static_managed.static.b04fd23c98500190 = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 1, ptr @_ZL14static_managed.managed, ptr @{{.+}}, i64 4, i64 4, ptr @_ZL14static_managed }, section "llvm_offload_entries"
// DEVICE-DAG: @_ZL14static_managed.static.b04fd23c98500190.managed = addrspace(1) externally_initialized global i32
static __managed__ int static_managed;

struct surfaceReference { int data; };
template <typename T, int Type>
struct __attribute__((device_builtin_surface_type)) surface : surfaceReference {};

// CHECK-DAG: @.offloading.entry.surf = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 2, ptr @surf, ptr @{{.+}}, i64 4, i64 2, ptr null }, section "llvm_offload_entries"
surface<void, 2> surf;
extern surface<void, 2> extern_surf;

struct textureReference { int data; };
template <typename T, int Type, int Normalized>
struct __attribute__((device_builtin_texture_type)) texture : textureReference {};

// CHECK-DAG: @.offloading.entry.tex = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 3, ptr @tex, ptr @{{.+}}, i64 4, i64 1, ptr null }, section "llvm_offload_entries"
// CIR-DAG: cu.var_registration = #cir.cu.var_registration<tex, Texture, texture_type = 1>
texture<float, 1, 0> tex;
// CHECK-DAG: @.offloading.entry.normalized = weak constant %{{.+}} { i64 0, i16 1, i16 4, i32 35, ptr @normalized, ptr @{{.+}}, i64 4, i64 2, ptr null }, section "llvm_offload_entries"
// CIR-DAG: cu.var_registration = #cir.cu.var_registration<normalized, Texture, normalized, texture_type = 2>
texture<float, 2, 1> normalized;
extern texture<float, 2, 1> extern_tex;

void *host_addresses[] = {&used_extern, &hidden};

int host_use() { return managed[0] + extern_managed + static_managed; }

__device__ int device_use() {
  return var + constant[0] + used_extern + hidden + managed[0] +
         extern_managed + static_managed;
}
