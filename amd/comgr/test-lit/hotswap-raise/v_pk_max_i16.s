; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=v_pk_max_i16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=PKMAX
; RUN: %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=v_pk_max3_i16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=PKMAX3
; RUN: %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=v_max3_i16_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MAX3
; RUN: %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=v_max3_i16_opsel_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MAX3OPSEL
;
; Lift tests for the signed i16 max family: the packed VOP3P forms
; v_pk_max_i16 / v_pk_max3_i16 and the true16 VOP3 scalar form v_max3_i16.
; Each pins that the opcode-map entry, CanonicalOp, and handler are wired up
; so the op lowers to the canonical llvm.smax intrinsic rather than refusing.
;
; PKMAX-LABEL: define amdgpu_kernel void @v_pk_max_i16_kernel(
; PKMAX: %pk_max_i16{{[0-9]*}} = call <2 x i16> @llvm.smax.v2i16(
; PKMAX: bitcast <2 x i16>
;
; PKMAX3-LABEL: define amdgpu_kernel void @v_pk_max3_i16_kernel(
; PKMAX3: %pk_max3_i16_m01{{[0-9]*}} = call <2 x i16> @llvm.smax.v2i16(
; PKMAX3: %pk_max3_i16{{[0-9]*}} = call <2 x i16> @llvm.smax.v2i16(
;
; MAX3-LABEL: define amdgpu_kernel void @v_max3_i16_kernel(
; MAX3: %vmax3_i16_m01{{[0-9]*}} = call i16 @llvm.smax.i16(
; MAX3: %vmax3_i16{{[0-9]*}} = call i16 @llvm.smax.i16(
;
; op_sel:[1,1,1,1] selects the high i16 of each source (lshr 16) and writes
; the high half of dst (shl 16 / merge_hi), exercising the true16 ternary
; op_sel decode and high-half merge shared with v_mad_u16.
; MAX3OPSEL-LABEL: define amdgpu_kernel void @v_max3_i16_opsel_kernel(
; MAX3OPSEL: lshr i32 %{{[^,]+}}, 16
; MAX3OPSEL: %vmax3_i16_m01{{[0-9]*}} = call i16 @llvm.smax.i16(
; MAX3OPSEL: %vmax3_i16{{[0-9]*}} = call i16 @llvm.smax.i16(
; MAX3OPSEL: shl i32 %{{[^,]+}}, 16
; MAX3OPSEL: %vmax3_i16_merge_hi{{[0-9]*}} = or i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_pk_max_i16_kernel
	.p2align	8
	.type	v_pk_max_i16_kernel,@function
v_pk_max_i16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_pk_max_i16 v2, v0, v1
	v_mov_b32_e32 v3, 0
	s_wait_loadcnt 0x0
	global_store_b32 v3, v2, s[0:1] scale_offset
	s_endpgm

	.globl	v_pk_max3_i16_kernel
	.p2align	8
	.type	v_pk_max3_i16_kernel,@function
v_pk_max3_i16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_pk_max3_i16 v3, v0, v1, v2
	v_mov_b32_e32 v4, 0
	s_wait_loadcnt 0x0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.globl	v_max3_i16_kernel
	.p2align	8
	.type	v_max3_i16_kernel,@function
v_max3_i16_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_max3_i16 v3, v0, v1, v2
	v_mov_b32_e32 v4, 0
	s_wait_loadcnt 0x0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.globl	v_max3_i16_opsel_kernel
	.p2align	8
	.type	v_max3_i16_opsel_kernel,@function
v_max3_i16_opsel_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_max3_i16 v3, v0, v1, v2 op_sel:[1,1,1,1]
	v_mov_b32_e32 v4, 0
	s_wait_loadcnt 0x0
	global_store_b32 v4, v3, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_max_i16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_pk_max3_i16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_max3_i16_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_max3_i16_opsel_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
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
    .name: v_pk_max_i16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_pk_max_i16_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_pk_max3_i16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_pk_max3_i16_kernel.kd
    .vgpr_count: 5
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_max3_i16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_max3_i16_kernel.kd
    .vgpr_count: 5
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: v_max3_i16_opsel_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_max3_i16_opsel_kernel.kd
    .vgpr_count: 5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
