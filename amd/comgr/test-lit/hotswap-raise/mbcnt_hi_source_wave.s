; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=mbcnt_hi_source_wave_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; A wave32 source `v_mbcnt_hi_u32_b32 src0, src1` is unconditionally a
; pass-through of src1: the hi-half mask `(1 << max(0, L - 32)) - 1` is
; empty for every source lane L in [0, 31].  Under wave32 -> wave64
; lifting the raw target `mbcnt.hi` would compute popcount(src0 &
; non_empty_mask) + src1 on target lanes 32..63 (which model
; source-wave-1 lanes 0..31), corrupting the source-wave-1 result and
; breaking Triton's `mbcnt_lo + mbcnt_hi` thread-id chain.  The lift
; therefore emits src1 directly when widening from wave32 source.

; The lift emits src1 directly under wave32 -> wave64; the user's
; v_mbcnt_hi has no IR footprint of its own, so the stored value flows
; straight from the v_mbcnt_lo lift (`mbcnt_lo_srcwave`) into the
; lane-active phi guarding the global_store.  The same-wave path
; would have produced a `%mbcnt_hi = call ... @llvm.amdgcn.mbcnt.hi`
; SSA value -- the cross-widen lift must NOT emit one.
;
; (The IR still contains `%lane_id = call ... @llvm.amdgcn.mbcnt.hi`
; sites: those come from the raiser's own `emitLaneId` helper, which
; the SPE wave-active checks use to compute target-lane ids, and are
; unrelated to the lifted user instruction.)

; CHECK-LABEL: define amdgpu_kernel void @mbcnt_hi_source_wave_kernel(
; CHECK: %mbcnt_lo_srcwave{{[0-9]*}} = add i32 %mbcnt_pop{{[0-9]*}}, 0
; CHECK-NOT: %mbcnt_hi{{[0-9]*}} = call i32 @llvm.amdgcn.mbcnt.hi
; CHECK: phi i32 [ %mbcnt_lo_srcwave{{[0-9]*}},

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	mbcnt_hi_source_wave_kernel
	.p2align	8
	.type	mbcnt_hi_source_wave_kernel,@function
mbcnt_hi_source_wave_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mbcnt_lo_u32_b32 v1, -1, 0
	v_mbcnt_hi_u32_b32 v1, -1, v1
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_hi_source_wave_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mbcnt_hi_source_wave_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         mbcnt_hi_source_wave_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
