; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=vcmp_masks_inactive_lanes_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; A plain V_CMP writes zero for inactive source lanes. The per-lane predicate
; must therefore be intersected with modeled EXEC before it is balloted into an
; SGPR destination or recorded in the cross-widening wave-mask shadow.
; CHECK-LABEL: define amdgpu_kernel void @vcmp_masks_inactive_lanes_kernel(
; CHECK: %vcmpf = fcmp ord float
; CHECK: %vcmp_active = and i1 %vcmpf, %{{[^ ]+}}
; CHECK: %vcmp_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %vcmp_active)
; CHECK: %wm_shadow_exec = call i64 @llvm.amdgcn.ballot.i64(i1 %vcmp_active)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl vcmp_masks_inactive_lanes_kernel
	.p2align 8
	.type vcmp_masks_inactive_lanes_kernel,@function
vcmp_masks_inactive_lanes_kernel:
	v_mov_b32_e32 v0, 0
	s_mov_b32 exec_lo, 1
	v_cmp_o_f32_e64 s2, v0, v0
	v_cndmask_b32_e64 v1, 0, 1, s2
	s_endpgm
.Lfunc_end:
	.size vcmp_masks_inactive_lanes_kernel, .Lfunc_end-vcmp_masks_inactive_lanes_kernel

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel vcmp_masks_inactive_lanes_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 3
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 64
    .name: vcmp_masks_inactive_lanes_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 3
    .symbol: vcmp_masks_inactive_lanes_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
