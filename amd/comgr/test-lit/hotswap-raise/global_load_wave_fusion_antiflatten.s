; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=global_load_antiflatten_kernel 2>/dev/null | %FileCheck %s

; Regression guard for the wave-native (wave32 source -> wave64 target) global
; load aperture fix.
;
; Under wave-native cross-widening the projection fuses two source wave32 into
; one target wave64 and seeds hardware EXEC = -1 via init_whole_wave. A global
; load guarded only by a single per-lane source-EXEC diamond (`spe_do` -> load
; -> phi) is a hammock the AMDGPU back-end if-converts: it drops the divergent
; branch and runs the load unconditionally under EXEC = -1. On a partial tail
; wave that pairs an in-bounds source wave with a source wave whose lanes are
; past the problem size, the unconditional load dereferences out-of-range
; addresses -> HSA aperture violation (allocation-dependent). See
; `emitMemOpUnderExecHardened` in handle-flat.cpp.
;
; The fix nests the load in a second, non-constant-foldable `lane_id < wave_size`
; branch (`memop_do` / `memop_cont`) inside the source-EXEC `spe_do` guard, so
; the back-end cannot collapse the hammock and must keep EXEC masking. This test
; asserts that structure is present for a wave-native global load. (The escape
; hatch HSA_HOTSWAP_DISABLE_LOAD_ANTIFLATTEN restores the plain diamond.)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	global_load_antiflatten_kernel
	.p2align	8
	.type	global_load_antiflatten_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @global_load_antiflatten_kernel(
global_load_antiflatten_kernel:
	global_load_b32 v1, v0, s[0:1]
	; The source-EXEC diamond opens, then the load is nested one level deeper in
	; an anti-if-conversion branch whose result feeds the memop_cont phi.
	; CHECK:      spe_do{{.*}}:
	; CHECK:        br i1 %{{.+}}, label %memop_do{{.*}}, label %memop_cont{{.*}}
	; CHECK:      memop_do{{.*}}:
	; CHECK:        %{{.+}} = load float, ptr addrspace(1)
	; CHECK:      memop_cont{{.*}}:
	; CHECK:        phi i32 [ %{{.+}}, %memop_do{{.*}} ]
	global_store_b32 v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_antiflatten_kernel
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           global_load_antiflatten_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         global_load_antiflatten_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
