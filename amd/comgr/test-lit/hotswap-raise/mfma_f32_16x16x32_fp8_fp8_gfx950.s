; Cross-target lift of a native fp8 MFMA: gfx950 (OCP fp8) source raised to
; gfx942 (FNUZ fp8). gfx950's v_mfma_f32_16x16x32_fp8_fp8 interprets its A/B
; bytes as OCP E4M3; gfx942's same-named MFMA reads FNUZ. handleMFMA must
; re-encode the A/B operands OCP->FNUZ so the gfx942 MFMA computes the same
; values. A same-target gfx950->gfx950 lift must NOT re-encode.

; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=mfma_fp8_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS

; RUN: %llvm_mc -mcpu=gfx950 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=mfma_fp8_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME

; CROSS-LABEL: define amdgpu_kernel void @mfma_fp8_kernel(
; Both A and B operands re-encoded OCP->FNUZ (per-byte vectorized, two i64
; operands -> four <4 x i32> -> <4 x i8> truncs), then the gfx942 fp8 MFMA.
; CROSS-COUNT-4: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>
; CROSS: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> %{{[^,]+}}, i32 0, i32 0, i32 0)

; SAME-LABEL: define amdgpu_kernel void @mfma_fp8_kernel(
; Same source/target fp8 format: emit the MFMA directly, no re-encode.
; SAME: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(
; SAME-NOT: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text
	.globl	mfma_fp8_kernel
	.p2align	8
	.type	mfma_fp8_kernel,@function
mfma_fp8_kernel:
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0
	v_mov_b32_e32 v4, 0
	v_mov_b32_e32 v5, 0
	v_mov_b32_e32 v6, 0
	v_mov_b32_e32 v7, 0
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[0:1], v[2:3], v[4:7]
	s_nop 8
	v_mov_b32_e32 v8, 0
	global_store_dwordx4 v8, v[4:7], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mfma_fp8_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 9
		.amdhsa_next_free_sgpr 2
		.amdhsa_accum_offset 12
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mfma_fp8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         mfma_fp8_kernel.kd
    .vgpr_count:     9
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
