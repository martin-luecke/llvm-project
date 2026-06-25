; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=cvt_enc_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=cvt_enc_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_enc_kernel
	.p2align	8
	.type	cvt_enc_kernel,@function
cvt_enc_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0x40400000
	v_mov_b32_e32 v2, 0x40a00000
; CROSS-LABEL: define amdgpu_kernel void @cvt_enc_kernel(
; CROSS-DAG: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(float %{{.+}}, float %{{.+}}, i32 0, i1 false)
; CROSS-DAG: call i32 @llvm.amdgcn.cvt.pk.bf8.f32(float %{{.+}}, float %{{.+}}, i32 0, i1 false)
; CROSS-DAG: %e4m3_ocp{{[0-9]*}} = select
; CROSS-DAG: %e5m2_ocp{{[0-9]*}} = select
; CROSS-DAG: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>
; SAME-LABEL: define amdgpu_kernel void @cvt_enc_kernel(
; SAME: call i32 @llvm.amdgcn.cvt.pk.fp8.f32(
; SAME-NOT: trunc <4 x i32> %{{[^ ]+}} to <4 x i8>
	v_cvt_pk_fp8_f32 v0, v1, v2
	v_cvt_pk_bf8_f32 v3, v1, v2
	v_mov_b32_e32 v5, 0
	s_wait_kmcnt 0x0
	global_store_b64 v5, v[0:1], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_enc_kernel
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
    .name:           cvt_enc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_enc_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
