; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx950 \
; RUN:     --emit-ir=buffer_store_antiflatten_kernel | %FileCheck %s

; Regression guard for the wave-native (wave32 source -> wave64 target) MUBUF
; buffer_store partial-tail OOB fix -- the STORE-side sibling of the #158
; buffer_load antiflatten (buffer_load_wave_fusion_antiflatten.s /
; emitMubufLoadUnderExecHardened) (rocm-systems#159, store-side third defect).
;
; Under wave-native cross-widening the projection fuses two source wave32 into
; one target wave64 and seeds hardware EXEC = -1 via init_whole_wave. The MUBUF
; store epilogue (the attention output write in the gemma prefill `_fwd_kernel`)
; was emitted WHOLE-WAVE (unconditional, at ambient EXEC = -1) to preserve the
; source packet's full-wave-issue + per-lane-OOB-suppression contract. But at a
; partial tail a target wave fuses an in-bounds source wave with a source wave
; whose lanes are past the problem size; the store-address VGPR is an EXEC-gated
; phi whose inactive arm holds a STALE prior real offset (a live wild ~4.28 GB
; per-lane offset, not the 0x80000000 OOB sentinel). NUM_RECORDS on the target
; descriptor is ~2 GB, so base+offset wraps into an unmapped page and the
; whole-wave store faults on those partial-tail lanes (rocgdb-pinned on the
; gemma prefill `_fwd_kernel` on gfx950: `buffer_store_dwordx4 ... offen`,
; exec = -1, lanes 0-31 offset 0xff823000/0xff825000, EXCP.MEM_VIOL).
;
; The fix nests the store in a non-constant-foldable
; `source_lane_active && lane_id < wave_size` branch (`mubuf_store_do` /
; `mubuf_store_cont`) so source-inactive partial-tail lanes (whose address phi
; holds the stale wild offset) skip the store, while every valid source lane
; still issues it -- without the `get_num_kv_splits_triton` over-masking a full
; per-lane emitUnderExec diamond caused. The `lane_id < wave_size` term is
; always-true-but-not-provably-so, so the back-end cannot if-convert the diamond
; back to an unconditional exec = -1 store.
;
; Option 1 (Martin review resolution): (1) the store predicate is the
; DISPATCH-TIME active bit (init_whole_wave's original per-lane mask), NOT the
; modeled `emitLaneActiveBit` that over-masked get_num_kv_splits_triton -- so
; every genuinely-dispatched lane still stores, only phantom partial-tail lanes
; are masked; (2) the guard is lowered through the projection-owned
; emitGuardedMemOp primitive (llvm.amdgcn.if/end.cf, SI_IF pseudo), a
; backend-respected contract, not the `lane_id < wave_size` tautology. No env
; escape hatch.
;
; Discriminator: post-fix the store is nested in `gmo_do` / `gmo_cont` under an
; llvm.amdgcn.if guarded by `dispatch_active`. Pre-fix (#158 base) the
; WaveNative store is emitted unconditionally (no guard block).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_store_antiflatten_kernel
	.p2align	8
	.type	buffer_store_antiflatten_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_store_antiflatten_kernel(
buffer_store_antiflatten_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 4, v0
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	v_mov_b32_e32 v4, 1
	v_mov_b32_e32 v5, 2
	v_mov_b32_e32 v6, 3
	v_mov_b32_e32 v7, 4
	s_wait_kmcnt 0x0
	; Option 1: the store is guarded on the DISPATCH-TIME active bit (the
	; init_whole_wave original mask), NOT the modeled EXEC that over-masked
	; get_num_kv_splits_triton, and lowered through the projection-owned
	; emitGuardedMemOp primitive. The dispatch-active predicate is emitted
	; (`dispatch_bit`); in this SYNTHETIC fully-dispatched wave the optimizer
	; can prove every lane active and folds the guard away (correct: no phantom
	; lane to mask) -- the discriminating partial-tail masking is exercised on
	; live gfx950, and the backend-respected lowering is proven in
	; buffer_load_exec_survives_lowered_isa.s.
	; CHECK:        dispatch_bit
	; CHECK:      gmo_do{{.*}}:
	; CHECK:        call void @llvm.amdgcn.raw.buffer.store.v4i32(
	; CHECK:      gmo_cont{{.*}}:
	buffer_store_b128 v[4:7], v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_antiflatten_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           buffer_store_antiflatten_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_antiflatten_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
