// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -fclangir -emit-llvm %s -o - | FileCheck %s \
// RUN:   --implicit-check-not=__hip_module_ctor \
// RUN:   --implicit-check-not=.offloading
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc \
// RUN:   -emit-llvm %s -o - | FileCheck %s \
// RUN:   --implicit-check-not=__hip_module_ctor \
// RUN:   --implicit-check-not=.offloading
// CHECK: target triple = "x86_64-unknown-linux-gnu"
