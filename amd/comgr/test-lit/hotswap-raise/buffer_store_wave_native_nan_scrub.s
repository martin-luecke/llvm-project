; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=buffer_store_wave_native_nan_scrub_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=buffer_store_wave_native_nan_scrub_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=MR
;
; Regression guard for the WaveNative NaN-scrub on MUBUF store data.
;
; Source program for kernels like Triton flash attention emits a
; `v_cmp_o_f32 + v_cndmask v_dst, 0, v_src, vcc` pair just before the
; final `buffer_store_*` to convert NaN values (produced by softmax's
; legitimate fully-masked-row 0/0 path) to 0 so that NaN does not
; propagate to global memory. The lifter's CFG walker drops this
; source instruction sequence because it sits in a fall-through block
; between two adjacent waterfall-loop labels — a code shape the
; current walker does not follow into. The result is that under
; WaveNative cross-widening (gfx1250 wave32 → gfx942 wave64) any NaN
; lane reaches global writeback unmasked, and the consuming model
; produces all-NaN logits.
;
; The fix in `handle-mubuf.cpp` mirrors the source's intent by
; scrubbing NaN per-element to 0 in the store-data VGPR immediately
; before the lifted `amdgcn.raw.buffer.store.*` call. This only
; applies when the projection holds the full-wave hardware EXEC
; invariant (`providesFullWaveExecInvariant()`) — i.e. WaveNative —
; because that is exactly the regime where the source's per-instruction
; EXEC-mask gating is defeated. Under ModuloReplication the source's
; EXEC mask still gates the store via the surrounding spe_skip diamond,
; so no scrub is needed and the IR stays unchanged.

; WaveNative: bitcast the store data to f32, fcmp uno self → NaN mask,
; select(IsNaN, 0, val) → buffer.store of the scrubbed value.
; WN-LABEL: define amdgpu_kernel void @buffer_store_wave_native_nan_scrub_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; WN: %[[F:.+]] = bitcast i32 %{{.+}} to float
; WN: %[[UNO:.+]] = fcmp uno float %[[F]], %[[F]]
; WN: %[[SCRUB:.+]] = select i1 %[[UNO]], i32 0, i32 %{{.+}}
; WN: call void @llvm.amdgcn.raw.buffer.store.i32(i32 %[[SCRUB]],

; ModuloReplication: no fcmp uno / NaN-scrub; the store stays inside a
; spe_skip diamond gated by source EXEC, which carries the source's
; per-lane predicate.
; MR-LABEL: define amdgpu_kernel void @buffer_store_wave_native_nan_scrub_kernel(
; MR-NOT: fcmp uno
; MR: br i1 %{{[^,]+}}, label %spe_do{{[0-9]*}}, label %spe_skip{{[0-9]*}}
; MR: spe_do{{[0-9]*}}:
; MR: call void @llvm.amdgcn.raw.buffer.store.i32(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_store_wave_native_nan_scrub_kernel
	.p2align	8
	.type	buffer_store_wave_native_nan_scrub_kernel,@function
buffer_store_wave_native_nan_scrub_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s2, 16
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	buffer_store_dword v0, v1, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_wave_native_nan_scrub_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
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
    .name:           buffer_store_wave_native_nan_scrub_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_wave_native_nan_scrub_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
