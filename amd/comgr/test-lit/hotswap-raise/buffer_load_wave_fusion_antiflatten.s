; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx950 \
; RUN:     --emit-ir=buffer_load_antiflatten_kernel | %FileCheck %s

; Regression guard for the wave-native (wave32 source -> wave64 target) MUBUF
; buffer_load aperture fix -- the buffer-path sibling of the global-load
; antiflatten (handle-flat.cpp emitMemOpUnderExecHardened).
;
; Under wave-native cross-widening the projection fuses two source wave32 into
; one target wave64 and seeds hardware EXEC = -1 via init_whole_wave. A
; buffer_load guarded only by a single per-lane source-EXEC diamond
; (`spe_do` -> raw.ptr.buffer.load -> phi) is a hammock the AMDGPU back-end
; if-converts: it drops the divergent branch and runs the load unconditionally
; under EXEC = -1. On a partial tail wave that pairs an in-bounds source wave
; with a source wave whose lanes are past the problem size, the unconditional
; load dereferences intentionally-out-of-range buffer offsets that the source
; masks off later (rocgdb-pinned on the gemma prefill `_fwd_kernel` on gfx950:
; `buffer_load_dwordx4 ... offen`, exec = -1, offset 0x3fe0000). The gfx942
; NUM_RECORDS is large enough that the hardware OOB clamp does not catch it.
;
; The fix nests the load in a second, non-constant-foldable
; `lane_id < wave_size` branch (`mubuf_memop_do` / `mubuf_memop_cont`) inside the
; source-EXEC `spe_do` guard, so the back-end cannot collapse the hammock and
; must keep EXEC masking. (Escape hatch: HSA_HOTSWAP_DISABLE_LOAD_ANTIFLATTEN.)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_load_antiflatten_kernel
	.p2align	8
	.type	buffer_load_antiflatten_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @buffer_load_antiflatten_kernel(
buffer_load_antiflatten_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 4, v0
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	; The source-EXEC diamond opens, then the buffer load is nested one level
	; deeper in an anti-if-conversion branch feeding the mubuf_memop_cont phi.
	; CHECK:      spe_do{{.*}}:
	; CHECK:        br i1 %{{.+}}, label %mubuf_memop_do{{.*}}, label %mubuf_memop_cont{{.*}}
	; CHECK:      mubuf_memop_do{{.*}}:
	; CHECK:        call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(
	; CHECK:      mubuf_memop_cont{{.*}}:
	buffer_load_b128 v[4:7], v0, s[0:3], null offen
	s_wait_loadcnt 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_load_antiflatten_kernel
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
    .name:           buffer_load_antiflatten_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_load_antiflatten_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
