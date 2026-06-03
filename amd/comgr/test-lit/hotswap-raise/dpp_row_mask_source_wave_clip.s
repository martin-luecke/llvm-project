; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=dpp_row_mask_source_wave_clip_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression guard for the DPP row_mask / bank_mask source-wave clip
; (commit 8b4799a38dc2 "Scope DPP row_mask/bank_mask to source wave
; under widening").
;
; v_mov_b32_dpp ... row_mask:N, bank_mask:M gates the destination
; write per source-wave row / bank, NOT per target-wave physical
; row / bank.  Under wave32 -> wave64 emulation a single target
; wave hosts two source waves on target lanes 0..31 and 32..63, so
; the row / bank index used for the gating select must be derived
; from `LaneId & (SourceWaveSize - 1)` (== `LaneId & 31` here),
; otherwise `row_mask:0x1` (source-row 0 only) would suppress
; target lanes 32..47 even though they ARE source-row 0 of the
; second source wave.
;
; This fixture exercises the gating path (non-0xF row_mask) and
; pins the IR signature of the clip: an AND of the target LaneId
; with 31, an LShr-by-4 to extract the source-wave row, an LShr-by-2
; to extract the source-wave bank, the row_mask / bank_mask shift-
; and-test predicates, and the final select that picks DppVal when
; the source-wave row AND bank are active and OldVal otherwise.
;
; The source-fetch side (within-row, row-base, ds_bpermute selector)
; stays target-physical by ISA definition (DPP source fetch is per-
; physical-row), so the within_row / row_base derivations remain
; based on the raw target LaneId.

; CHECK-LABEL: define amdgpu_kernel void @dpp_row_mask_source_wave_clip_kernel(

; The faithful-lift update.dpp.i32 call MUST NOT survive the
; cross-widening rewrite.
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(

; Source-fetch side: target-physical within-row / row-base.
; CHECK-DAG: %cwd_dpp_within_row = and i32 %{{.+}}, 15
; CHECK-DAG: %cwd_dpp_row_base = and i32 %{{.+}}, -16

; Destination-gate side: clip target LaneId to the source-wave-local
; range (wave32 source => mask 31), then derive the source-wave row
; and bank from the clipped lane.
; CHECK-DAG: %cwd_dpp_source_lane = and i32 %{{.+}}, 31
; CHECK-DAG: %[[SROW_SH:.+]] = lshr i32 %cwd_dpp_source_lane, 4
; CHECK-DAG: %cwd_dpp_source_row = and i32 %[[SROW_SH]], 3
; CHECK-DAG: %[[SBANK_SH:.+]] = lshr i32 %cwd_dpp_source_lane, 2
; CHECK-DAG: %cwd_dpp_source_bank = and i32 %[[SBANK_SH]], 3

; row_mask = 0x1 -> shift by SourceRow, low-bit test.
; CHECK-DAG: %[[RM_SH:.+]] = lshr i32 1, %cwd_dpp_source_row
; CHECK-DAG: %[[RM_BIT:.+]] = and i32 %[[RM_SH]], 1
; CHECK-DAG: %cwd_dpp_row_active = icmp ne i32 %[[RM_BIT]], 0

; bank_mask = 0xF -> shift by SourceBank, low-bit test.
; CHECK-DAG: %[[BM_SH:.+]] = lshr i32 15, %cwd_dpp_source_bank
; CHECK-DAG: %[[BM_BIT:.+]] = and i32 %[[BM_SH]], 1
; CHECK-DAG: %cwd_dpp_bank_active = icmp ne i32 %[[BM_BIT]], 0

; Combined lane-active predicate gates DppVal vs OldVal.
; CHECK-DAG: %cwd_dpp_lane_active = and i1 %cwd_dpp_row_active, %cwd_dpp_bank_active
; CHECK-DAG: %cwd_dpp_gated = select i1 %cwd_dpp_lane_active, i32 %{{.+}}, i32 %{{.+}}

; The bpermute call lowers the source-fetch path (DAG: the call
; emits earlier in the basic block than the gate select).
; CHECK-DAG: %cwd_dpp_bperm = call i32 @llvm.amdgcn.ds.bpermute(i32 %cwd_dpp_selector, i32 %{{[^,]+}})

; The bpermute intrinsic declaration survives in the module.
; CHECK: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	dpp_row_mask_source_wave_clip_kernel
	.p2align	8
	.type	dpp_row_mask_source_wave_clip_kernel,@function
dpp_row_mask_source_wave_clip_kernel:   ; @dpp_row_mask_source_wave_clip_kernel
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
	v_mad_u32 v0, s0, s4, v0
	global_load_b32 v1, v0, s[2:3] scale_offset
	s_wait_loadcnt 0x0
	;;#ASMSTART
	v_mov_b32_dpp v1, v1 row_shr:1 row_mask:0x1 bank_mask:0xf bound_ctrl:1

	;;#ASMEND
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel dpp_row_mask_source_wave_clip_kernel
		.amdhsa_kernarg_size 264
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
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
    .max_flat_workgroup_size: 1024
    .name:           dpp_row_mask_source_wave_clip_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         dpp_row_mask_source_wave_clip_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
