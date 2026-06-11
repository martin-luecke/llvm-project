; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=andn2_saveexec_kernel | %FileCheck %s
;
; S_AND_NOT1_SAVEEXEC_B32 (canonical S_ANDN2_SAVEEXEC_B32) lifts EXEC to
; `Src & ~OldExec`. See the SAVEEXEC handlers in handle-sop1.cpp.

; CHECK-LABEL: define amdgpu_kernel void @andn2_saveexec_kernel(
; New EXEC = Src & ~OldExec: the inverted operand is the saved EXEC.
; CHECK: %[[NOTEXEC:[0-9]+]] = xor i64 %saved_exec, -1
; CHECK: %new_exec = and i64 %{{[a-zA-Z0-9_.]+}}, %[[NOTEXEC]]

        .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
        .amdhsa_code_object_version 6
        .text
        .globl  andn2_saveexec_kernel
        .p2align        8
        .type   andn2_saveexec_kernel,@function
andn2_saveexec_kernel:
        v_cmp_gt_f32_e64 s2, v0, v1
        s_and_not1_saveexec_b32 s3, s2
        v_mov_b32 v4, 1
        ds_store_b32 v5, v4
        s_or_b32 exec_lo, exec_lo, s3
        s_endpgm
        .section        .rodata,"a",@progbits
        .p2align        6, 0x0
        .amdhsa_kernel andn2_saveexec_kernel
                .amdhsa_kernarg_size 0
                .amdhsa_user_sgpr_count 0
                .amdhsa_next_free_vgpr 6
                .amdhsa_next_free_sgpr 8
                .amdhsa_wavefront_size32 1
                .amdhsa_float_denorm_mode_32 3
                .amdhsa_inst_pref_size 1
        .end_amdhsa_kernel
        .text
        .p2alignl 7, 3214868480
        .fill 96, 4, 3214868480
        .amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           andn2_saveexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         andn2_saveexec_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
        .end_amdgpu_metadata
