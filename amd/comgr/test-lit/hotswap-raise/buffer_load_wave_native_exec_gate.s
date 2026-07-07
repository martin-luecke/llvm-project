; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=buffer_load_wave_native_exec_gate_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=buffer_load_wave_native_exec_gate_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MR
;
; Regression guard for rocm-systems#148: a MUBUF *load* under WaveNative
; gfx1250->gfx942 translation must be EXEC-gated (wrapped in an
; `emitUnderExec` diamond), so source-inactive / WaveNative "phantom" lanes
; never issue the load.
;
; The Gemma prefill-attention `_fwd_kernel` computes a per-lane K/V-cache
; voffset with the Triton masked-load idiom `v_cndmask v, 0x80000000, v, sN`
; (valid lane -> real offset, invalid -> OOB sentinel) and then does
; `buffer_load ... offen`.  Before this fix the raiser emitted the buffer
; load on *every* target lane; a phantom lane whose offset VGPR held a stale
; reused value (a softmax float, not the 0x80000000 sentinel) combined with
; the NUM_RECORDS remap (0x00ffffff -> 0x7ffffffe in mubuf-addr.cpp) landed
; in-bounds on the target and dereferenced a wild address -> TCP VM page
; fault.  On real wave32 hardware EXEC masks the load; the WaveNative model
; must reproduce that with an explicit diamond.
;
; This mirrors the GLOBAL_LOAD fix in handle-flat.cpp (2026-04-22); loads are
; always safe to gate (a masked-out lane's loaded value is discarded), unlike
; the masked-store path (buffer_store_wave_native_oob_mask.s) which stays
; full-wave under WaveNative.
;
; Both projections must gate the load: WaveNative wraps it in a diamond even
; though it holds a full-wave hardware EXEC invariant for stores, because a
; load has no per-lane-OOB-offset contract to preserve.

; The buffer load must be EXEC-gated: emitted inside an `emitUnderExec`
; `spe_do` block, with its result merged by a phi that yields the loaded
; value on active lanes and `undef` on the skipped (inactive / phantom)
; lanes. The phi's `[ %buf_ld_rawptr, %spe_do* ], [ undef, %spe_skip* ]`
; shape is the unambiguous proof the load is conditional. Both projections
; must gate the load (loads have no per-lane-OOB-offset contract to
; preserve, unlike the masked-store path in
; buffer_store_wave_native_oob_mask.s).
;
; WN-LABEL: define amdgpu_kernel void @buffer_load_wave_native_exec_gate_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; WN: %buf_ld_rawptr = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(
; WN: = phi i32 [ %buf_ld_rawptr, %spe_do{{[0-9]*}} ], [ undef, %spe_skip{{[0-9]*}} ]

; MR-LABEL: define amdgpu_kernel void @buffer_load_wave_native_exec_gate_kernel(
; MR: %buf_ld_rawptr = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(
; MR: = phi i32 [ %buf_ld_rawptr, %spe_do{{[0-9]*}} ], [ undef, %spe_skip{{[0-9]*}} ]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_load_wave_native_exec_gate_kernel
	.p2align	8
	.type	buffer_load_wave_native_exec_gate_kernel,@function
buffer_load_wave_native_exec_gate_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v2, 64
	v_mov_b32_e32 v3, 0
	v_cmp_eq_u32_e32 vcc_lo, 0, v0
	v_cndmask_b32_e32 v1, v2, v3, vcc_lo
	s_mov_b32 s2, 4
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	buffer_load_dword v4, v1, s[0:3], null offen
	s_wait_loadcnt 0
	buffer_store_dword v4, v1, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_load_wave_native_exec_gate_kernel
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
    .max_flat_workgroup_size: 64
    .name:           buffer_load_wave_native_exec_gate_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_load_wave_native_exec_gate_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
