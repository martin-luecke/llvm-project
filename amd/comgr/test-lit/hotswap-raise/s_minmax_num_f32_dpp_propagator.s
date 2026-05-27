; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_max_num_f32_dpp_propagator_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MAX
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:   --emit-ir=s_min_num_f32_dpp_propagator_kernel 2>&1 \
; RUN: | %FileCheck %s --check-prefix=MIN
;
; DPP-to-readfirstlane chains preserve per-source-wave identity by rewriting the
; DPP and readfirstlane sites.  Scalar NUM extrema lower to llvm.maximumnum /
; llvm.minimumnum; the forward-use classifier must keep walking through those
; calls just like maxnum/minnum instead of refusing them as unaudited intrinsics.

; MAX-LABEL: define amdgpu_kernel void @s_max_num_f32_dpp_propagator_kernel(
; MAX-NOT: call {{.*}}@llvm.amdgcn.update.dpp.i32
; MAX: call i32 @llvm.amdgcn.ds.bpermute
; MAX: call i32 @llvm.amdgcn.ds.bpermute
; MAX: call float @llvm.maximumnum.f32(
; MAX-NOT: call {{.*}}@llvm.amdgcn.update.dpp.i32
; MAX-NOT: call {{.*}}@llvm.amdgcn.readfirstlane

; MIN-LABEL: define amdgpu_kernel void @s_min_num_f32_dpp_propagator_kernel(
; MIN-NOT: call {{.*}}@llvm.amdgcn.update.dpp.i32
; MIN: call i32 @llvm.amdgcn.ds.bpermute
; MIN: call i32 @llvm.amdgcn.ds.bpermute
; MIN: call float @llvm.minimumnum.f32(
; MIN-NOT: call {{.*}}@llvm.amdgcn.update.dpp.i32
; MIN-NOT: call {{.*}}@llvm.amdgcn.readfirstlane

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_max_num_f32_dpp_propagator_kernel
	.p2align	8
	.type	s_max_num_f32_dpp_propagator_kernel,@function
s_max_num_f32_dpp_propagator_kernel:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_mov_b32 s5, 0x3f800000
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_mov_b32_dpp v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_readfirstlane_b32 s4, v1
	s_max_num_f32 s4, s4, s5
	;;#ASMEND
	v_mov_b32_e32 v1, s4
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

	.globl	s_min_num_f32_dpp_propagator_kernel
	.p2align	8
	.type	s_min_num_f32_dpp_propagator_kernel,@function
s_min_num_f32_dpp_propagator_kernel:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_mov_b32 s5, 0x3f800000
	s_wait_kmcnt 0x0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_mov_b32_dpp v1, v1 row_shr:4 row_mask:0xf bank_mask:0xf bound_ctrl:1
	v_readfirstlane_b32 s4, v1
	s_min_num_f32 s4, s4, s5
	;;#ASMEND
	v_mov_b32_e32 v1, s4
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_max_num_f32_dpp_propagator_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel s_min_num_f32_dpp_propagator_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           s_max_num_f32_dpp_propagator_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         s_max_num_f32_dpp_propagator_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           s_min_num_f32_dpp_propagator_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         s_min_num_f32_dpp_propagator_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
