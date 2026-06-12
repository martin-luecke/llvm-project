; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=s_cbranch_vcc_wave_mask_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; s_cbranch_vcc{z,nz} is a scalar branch on whether the whole VCC wave mask is
; zero. It must not branch on the current lane's i1 VCC bit.
;
; CHECK-LABEL: define amdgpu_kernel void @s_cbranch_vcc_wave_mask_kernel(
; CHECK: %vcc_ballot{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(
; CHECK: %vcc_is_zero{{[0-9]*}} = icmp eq i64 %vcc_ballot{{[0-9]*}}, 0
; CHECK: br i1 %vcc_is_zero

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_cbranch_vcc_wave_mask_kernel
	.p2align	8
	.type	s_cbranch_vcc_wave_mask_kernel,@function
s_cbranch_vcc_wave_mask_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	s_cbranch_vccz .Lzero
	v_mov_b32_e32 v1, 1
	s_branch .Lstore
.Lzero:
	v_mov_b32_e32 v1, 0
.Lstore:
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_cbranch_vcc_wave_mask_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: s_cbranch_vcc_wave_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: s_cbranch_vcc_wave_mask_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
