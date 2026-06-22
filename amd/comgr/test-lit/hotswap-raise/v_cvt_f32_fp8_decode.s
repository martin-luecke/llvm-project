; fp8 decode lift: v_cvt_pk_f32_fp8 / v_cvt_f32_fp8 from a gfx1250 (OCP fp8)
; source to a gfx942 (FNUZ fp8) target. The in-register fp8 bytes are OCP;
; gfx942's hw decode reads FNUZ, so the decoder INPUT byte is re-encoded
; OCP->FNUZ before the hw cvt. A same-target gfx1250->gfx1250 lift must NOT
; re-encode (it emits the native decode directly).

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME

; CROSS-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; Each decoder input is re-encoded OCP->FNUZ (ending in a <4 x i32> -> <4 x i8>
; trunc) and the re-encoded dword feeds the hw decode intrinsic.
; CROSS-DAG: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 false)
; CROSS-DAG: call float @llvm.amdgcn.cvt.f32.fp8(i32 %{{[^,]+}}, i32 0)
; CROSS-DAG: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>

; SAME-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; Same source/target fp8 format: native decode, no re-encode.
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(
; SAME-NOT: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_dec_kernel
	.p2align	8
	.type	cvt_dec_kernel,@function
cvt_dec_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x40404040
	v_cvt_pk_f32_fp8 v[2:3], v0
	v_cvt_f32_fp8 v4, v0
	v_mov_b32_e32 v5, 0
	s_wait_kmcnt 0x0
	global_store_b96 v5, v[2:4], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_dec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 6
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           cvt_dec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_dec_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
