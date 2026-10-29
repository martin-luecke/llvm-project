// RUN: echo -n HIP_OFFLOAD_OBJECT > %t.bin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -fclangir -fembed-offload-object=%t.bin -emit-llvm %s -o - \
// RUN:   | FileCheck %s --implicit-check-not=__hip_module_ctor
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -fembed-offload-object=%t.bin -emit-llvm %s -o - \
// RUN:   | FileCheck %s --implicit-check-not=__hip_module_ctor
// CHECK: @llvm.embedded.object = private constant [18 x i8] c"HIP_OFFLOAD_OBJECT", section ".llvm.offloading", align 8, !exclude
// CHECK: @llvm.compiler.used = {{.+}}ptr @llvm.embedded.object
// CHECK: !llvm.embedded.objects = !{![[EMBED:.+]]}
// CHECK: ![[EMBED]] = !{ptr @llvm.embedded.object, !".llvm.offloading"}

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -fclangir -fembed-offload-object=%t.bin -emit-obj %s -o %t.o
// RUN: llvm-readelf -S %t.o | FileCheck %s --check-prefix=OBJECT
// OBJECT: .llvm.offloading

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -fclangir -foffload-include-binary %t.missing -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=RAW --implicit-check-not=.hip_fatbin \
// RUN:   --implicit-check-not=__hip_module_ctor
// RAW: @.offloading.entry.var

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -fclangir -fembed-offload-object=%t.missing -emit-llvm %s -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=MISSING
// MISSING: error: could not open '{{.+}}.missing' for embedding

// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -fgpu-rdc \
// RUN:   -fclangir -emit-llvm /dev/null -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CUDA
// CUDA: error: CUDA RDC is not implemented

__attribute__((device)) int var;
