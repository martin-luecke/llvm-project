; A workitem.id.y()-derived divergent predicate (the reduce_kernel aperture
; class) whose source block is already 1024 threads. Doubling it would need 2048
; threads/block, past the gfx942 hardware max, so the scaled route is ineligible
; and the refusal stands. See hotswap/docs/modrep-predicate-chain.md sec. 10.4.

; Default (auto-upgrade): the size gate makes it ineligible, so the principled
; WaveNative y/z refusal stands rather than silently doubling past the max.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=scaled_modrep_too_large_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=NOUPGRADE
; NOUPGRADE: workitem.id.y()/.z()-derived predicate under WaveNative
; NOUPGRADE-NOT: ScaledModuloReplicationProjection
; NOUPGRADE-NOT: define amdgpu_kernel void @scaled_modrep_too_large_refuse_kernel(

; --force-scaled-modrep: refuse with the size-gate diagnostic.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && not raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_too_large_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SIZE
; SIZE: ScaledModuloReplicationProjection needs to launch 2048 thread
; SIZE-SAME: target hardware limit is 1024
; SIZE-NOT: define amdgpu_kernel void @scaled_modrep_too_large_refuse_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	scaled_modrep_too_large_refuse_kernel
	.type	scaled_modrep_too_large_refuse_kernel,@function
scaled_modrep_too_large_refuse_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	v_bfe_u32 v2, v0, 10, 10
	s_wait_kmcnt 0x0
	v_cmp_lt_u32_e64 s4, v2, 16
	v_cndmask_b32_e64 v0, -1, v0, s4
	v_mov_b32_e32 v1, v2
	global_store_b32 v1, v0, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scaled_modrep_too_large_refuse_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  1024
    .name:                     scaled_modrep_too_large_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         scaled_modrep_too_large_refuse_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
