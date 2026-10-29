// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-cir -o - %s | FileCheck --check-prefix=CIR-DEVICE %s
// RUN: %clang_cc1 -fclangir -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=LLVM-DEVICE %s
// RUN: %clang_cc1 -std=c++11 -fcuda-is-device -triple nvptx64-nvidia-cuda -emit-llvm -o - %s | FileCheck --check-prefix=OGCG-DEVICE %s
// RUN: %clang_cc1 -fclangir -clangir-disable-passes -std=c++11 \
// RUN:   -triple x86_64-unknown-linux-gnu -x hip -fgpu-rdc -cuid=abc \
// RUN:   -emit-cir -o %t.cir %s
// RUN: cir-opt %t.cir --verify-roundtrip | FileCheck %s --check-prefix=HOST

struct textureReference {
  int desc;
};

enum ReadMode {
  ElementType = 0,
  NormalizedFloat = 1
};

template <typename T, int dim = 1, ReadMode mode = ElementType>
struct __attribute__((device_builtin_texture_type)) texture
    : public textureReference {};

// HOST-DAG: cu.var_registration = #cir.cu.var_registration<tex, Texture, normalized, texture_type = 2>
texture<float, 2, NormalizedFloat> tex;
// HOST-DAG: cu.var_registration = #cir.cu.var_registration<element, Texture, texture_type = 1>
texture<float> element;

// CIR-DEVICE: cir.global external target_address_space(1) @tex = #cir.undef : !cir.cuda_texture

// CIR now matches OG CodeGen and emits undef for CUDA shadow variables.
// LLVM-DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef
// OGCG-DEVICE: @tex ={{.*}} addrspace(1) externally_initialized global i64 undef
