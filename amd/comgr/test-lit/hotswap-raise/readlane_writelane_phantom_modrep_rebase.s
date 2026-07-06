; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=readlane_writelane_phantom_modrep_rebase_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=REWRITE
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --disable-writelane-rewrite \
; RUN:     --emit-ir=readlane_writelane_phantom_modrep_rebase_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=UNCHANGED
;
; Regression fence for issue #146: `v_readlane_b32` / `v_writelane_b32`
; must be source-wave-rebased under the phantom-lane
; ModuloReplicationProjection regime, not just under ThreadLoopProjection.
;
; The kernel's `.max_flat_workgroup_size: 32` forces the phantom-lane
; MODREP fallback on the gfx942 (wave64) target (32 < 64). Under MODREP
; the raiser handler (handle-valu-cross-lane.cpp) leaves the native
; `@llvm.amdgcn.readlane` / `@llvm.amdgcn.writelane` in place (its
; source-wave rebase is gated on `sourceWaveScopedLaneOps()`, true only
; for ThreadLoopProjection). The correctness comes from the default-on
; post-raise `rewriteCrossLaneDivergent` pass, which rebases every
; cross-widen read/writelane symmetrically (behind the SGPR-forced
; use-chain safety net). This fixture pins that the DEFAULT path rebases
; here -- i.e. #146's silent wave32->wave64 miscompile cannot occur
; without explicitly opting out via `--disable-writelane-rewrite`.
;
; REWRITE path (default -- pass on):
;   * writelane -> `select ((lane_id & (W_s-1)) == lane_idx), val, old`
;   * readlane  -> `ds_bpermute(((lane_id & ~(W_s-1)) | lane_idx) << 2, src)`
;   both keyed on the canonical `cwd_lane_id_lo` + `cwd_lane_id` two-step
;   mbcnt lane-id built once at the entry block.
; REWRITE-LABEL: define amdgpu_kernel void @readlane_writelane_phantom_modrep_rebase_kernel(
; REWRITE: %cwd_lane_id_lo = call i32 @llvm.amdgcn.mbcnt.lo
; REWRITE: %cwd_lane_id = call i32 @llvm.amdgcn.mbcnt.hi
; REWRITE: %cwd_rl_selector = shl
; REWRITE: %cwd_readlane_rewritten = call i32 @llvm.amdgcn.ds.bpermute
; REWRITE: %cwd_wl_mask = icmp eq
; REWRITE: %cwd_writelane_rewritten = select i1
; REWRITE-NOT: call i32 @llvm.amdgcn.readlane
; REWRITE-NOT: call i32 @llvm.amdgcn.writelane
;
; UNCHANGED path (`--disable-writelane-rewrite` pins the pre-rewrite
; native form -- this is the shape #146 warns is a silent miscompile if
; it were ever the default):
; UNCHANGED-LABEL: define amdgpu_kernel void @readlane_writelane_phantom_modrep_rebase_kernel(
; UNCHANGED: call i32 @llvm.amdgcn.readlane
; UNCHANGED: call i32 @llvm.amdgcn.writelane
; UNCHANGED-NOT: cwd_readlane_rewritten
; UNCHANGED-NOT: cwd_writelane_rewritten

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	readlane_writelane_phantom_modrep_rebase_kernel
	.p2align	8
	.type	readlane_writelane_phantom_modrep_rebase_kernel,@function
readlane_writelane_phantom_modrep_rebase_kernel:
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
	v_mov_b32_e32 v1, 0
	;;#ASMSTART
	v_readlane_b32 s0, v1, 5
	
	;;#ASMEND
	;;#ASMSTART
	v_writelane_b32 v1, s0, 7
	
	;;#ASMEND
	v_xor_b32_e32 v1, s0, v1
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel readlane_writelane_phantom_modrep_rebase_kernel
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
    .max_flat_workgroup_size: 32
    .name:           readlane_writelane_phantom_modrep_rebase_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         readlane_writelane_phantom_modrep_rebase_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
