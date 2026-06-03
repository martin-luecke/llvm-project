; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=maximumnum_minimumnum_vgpr_safe_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WN
;
; Regression guard for classifying `llvm.maximumnum` / `llvm.minimumnum`
; as VGPR-safe propagators in the cross-lane-divergent rewriter.
;
; Source kernel chains a divergent `v_writelane_b32` -> `v_readfirstlane_b32`
; -> `s_max_num_f32` -> `s_min_num_f32` -> global store. The two
; `s_{max,min}_num_f32` SOP2 forms lift to `llvm.maximumnum.f32` /
; `llvm.minimumnum.f32` (the IEEE-754 2019 number-favoring max/min;
; see `s_minmax_num_f32.s` for the per-opcode lift contract). Under
; WaveNative the rewriter must:
;   1. Replace the `amdgcn.writelane` with the `cwd_writelane_rewritten`
;      select.
;   2. Convert the explicit `amdgcn.readfirstlane` into a source-wave
;      `ds_bpermute` broadcast (`readfirstlane_srcwave`).
;   3. Continue the forward use-chain walk *through* the
;      `llvm.maximumnum` / `llvm.minimumnum` intrinsic calls without
;      refusing (they meet the VGPR-safe-propagator bar -- same per-lane
;      VALU expansion as `maxnum` / `minnum`, only NaN handling
;      differs).
;
; Pre-patch (before commit 174c0784): the classifier saw the two
; intrinsics as Unknown, returned SGPRForced, and refused the rewrite
; for the whole function. This test pins that they are now handled.

; WN-NOT: refused
; WN-NOT: SGPR-forced
; WN-NOT: ThreadLoopProjection
; WN-LABEL: define amdgpu_kernel void @maximumnum_minimumnum_vgpr_safe_kernel(
; WN: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; WN: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; WN: %cwd_writelane_rewritten = select i1
; WN: %readfirstlane_srcwave = call i32 @llvm.amdgcn.ds.bpermute
; WN: call float @llvm.maximumnum.f32(
; WN: call float @llvm.minimumnum.f32(
; WN-NOT: call i32 @llvm.amdgcn.writelane
; WN-NOT: call i32 @llvm.amdgcn.readfirstlane

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	maximumnum_minimumnum_vgpr_safe_kernel
	.p2align	8
	.type	maximumnum_minimumnum_vgpr_safe_kernel,@function
maximumnum_minimumnum_vgpr_safe_kernel: ; @maximumnum_minimumnum_vgpr_safe_kernel
; %bb.0:
	s_clause 0x1
	s_load_b128 s[4:7], s[0:1], 0x0
	s_load_b32 s8, s[0:1], 0x1c
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s9, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s8, s8, 0xffff
	s_cmp_eq_u32 s9, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v0, s0, s8, v0
	;;#ASMSTART
	s_bfe_u32 s0, ttmp8, 0x50019

	;;#ASMEND
	;;#ASMSTART
	v_writelane_b32 v1, s0, 0

	;;#ASMEND
	;;#ASMSTART
	v_readfirstlane_b32 s2, v1

	;;#ASMEND
	;;#ASMSTART
	s_max_num_f32 s3, s2, s6
	s_min_num_f32 s8, s2, s7

	;;#ASMEND
	v_mov_b32_e32 v1, s3
	v_mov_b32_e32 v2, s8
	global_store_b32 v0, v1, s[4:5] scale_offset
	global_store_b32 v0, v2, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel maximumnum_minimumnum_vgpr_safe_kernel
		.amdhsa_kernarg_size 272
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 10
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
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:        12
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         20
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         28
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         30
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         32
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         34
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         36
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         38
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         80
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 272
    .max_flat_workgroup_size: 1024
    .name:           maximumnum_minimumnum_vgpr_safe_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         maximumnum_minimumnum_vgpr_safe_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
