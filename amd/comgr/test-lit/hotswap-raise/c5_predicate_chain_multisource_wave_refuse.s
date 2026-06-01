; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_multisource_wave_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=STDERR
;
; Regression fence for the multi-source-wave C5 refusal under
; WaveNativeProjection (the post-graduation default route, no
; `--disable-wave-native`).
;
; Background. WaveNativeProjection packs several source wave32 waves
; into one target wave64. A `workitem.id.x()` -> `icmp` against a
; compile-time constant `K` in `(0, W_s-1]` is a lane-position-scoped
; predicate that is source-wave-scoped: lane L of source wave 0 and
; lane L of source wave 1 share the same target lane-position
; (`lane_id MOD W_s`) once they are packed into the same target wave,
; so the same `icmp ult tid, K` evaluates the same on both source
; waves' lane L despite the two source waves having distinct `tid`
; values in the source ISA. This was observed as a runtime VM fault
; in DeepSeek-V3's fused_moe_kernel and is the motivating
; counterexample for extending the C5 classifier's WaveNative refusal
; from the phantom-lane regime (`wg < target_wave`) to also cover the
; multi-source-wave regime (`wg > source_wave`).
;
; Fixture shape. The kernel is the same `tid -> icmp ult, 16` C5
; shape as `c5_predicate_chain_tid.s`, with one change in the HSACO
; metadata: `max_flat_workgroup_size: 64`. With source wave 32 and
; target wave 64, `64 > source_wave=32` AND `64 >= target_wave=64`,
; so the launch packs two source wave32 waves into a single target
; wave64. Under the old gate the classifier let this through (the
; phantom-lane predicate `wg < target_wave` was the only WaveNative
; refusal); under the new gate it refuses.
;
; Pairing. `c5_predicate_chain_phantom_lane.s` pins the phantom-lane
; regime (wg < target_wave) -- the OTHER half of the WaveNative
; refusal. `c5_predicate_chain_tid.s`'s IR_WN line pins the
; `wg == 0 (unknown)` non-refusal contract by way of the suppression
; reason. Together the three fixtures fence the WaveNative C5
; refusal decision procedure on the three workgroup-size regimes
; the classifier can distinguish.

; Outer raise_cli failure line, named for coverage bucketing.
; STDERR: raise_cli: kernel 'c5_predicate_chain_multisource_wave_refuse_kernel' failed to raise:
; STDERR-SAME: cross-wave-predicate-chain

; The C5 classifier names the workitem.id.x-derived chain and the
; failing compile-time constant in the per-site refusal detail. The
; multi-source-wave path uses the same `formatRefusalDetail`
; rendering as the MODREP path, so the icmp predicate and the
; offending constant must appear in stderr.
; STDERR: transpiler: pre-translation abort:
; STDERR-SAME: cross-wave-predicate-chain
; STDERR-SAME: workitem.id.x-predicate-chain-classifier
; STDERR: icmp ult
; STDERR-SAME: compile-time constant 16
; STDERR-SAME: W_s-1=31
; STDERR: outcome: (c) refuse
; STDERR-SAME: WorkitemIdPredicateChain
; STDERR-SAME: Class 5

; Negative guard: this is NOT the phantom-lane regime. The
; phantom-lane diagnostic prefix must not appear on the
; multi-source-wave refusal -- if it does, the gate is mis-attributing
; the refusal cause and the operator-facing message would point at
; the wrong evidence.
; STDERR-NOT: phantom-lane regime
; STDERR-NOT: init_whole_wave

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c5_predicate_chain_multisource_wave_refuse_kernel
	.p2align	8
	.type	c5_predicate_chain_multisource_wave_refuse_kernel,@function
c5_predicate_chain_multisource_wave_refuse_kernel: ; @c5_predicate_chain_multisource_wave_refuse_kernel
; %bb.0:
	s_clause 0x1
	s_load_b32 s4, s[0:1], 0x14
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_xcnt 0x0
	s_bfe_u32 s0, ttmp6, 0x4000c
	s_and_b32 s1, ttmp6, 15
	s_add_co_i32 s0, s0, 1
	s_getreg_b32 s5, hwreg(HW_REG_IB_STS2, 6, 4)
	s_mul_i32 s0, ttmp9, s0
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)
	s_add_co_i32 s1, s1, s0
	s_wait_kmcnt 0x0
	s_and_b32 s4, s4, 0xffff
	s_cmp_eq_u32 s5, 0
	s_cselect_b32 s0, ttmp9, s1
	v_mad_u32 v1, s0, s4, v0
	;;#ASMSTART
	v_cmp_lt_u32_e64 s0, v0, 16

	;;#ASMEND
	;;#ASMSTART
	v_cndmask_b32_e64 v0, -1, v0, s0

	;;#ASMEND
	global_store_b32 v1, v0, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c5_predicate_chain_multisource_wave_refuse_kernel
		.amdhsa_kernarg_size 264
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
      - .offset:         8
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         12
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         16
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         20
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         22
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         24
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         26
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         28
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         30
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         48
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         56
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         72
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 264
    .max_flat_workgroup_size: 64
    .name:           c5_predicate_chain_multisource_wave_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c5_predicate_chain_multisource_wave_refuse_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
