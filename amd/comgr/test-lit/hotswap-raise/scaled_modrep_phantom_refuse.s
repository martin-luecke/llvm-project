; ScaledModuloReplicationProjection (the scaled dispatch) is invalid in the
; phantom-lane regime (max_flat_workgroup_size < target wave size): it drops the
; MODREP phantom-lane clamp, so hardware lanes past the scaled block size would
; address out of bounds. The auto-upgrade never reaches a sub-wave kernel (those
; route to plain MODREP), so this only guards an explicit --force-scaled-modrep;
; that forced route must REFUSE rather than mislower. See
; hotswap/docs/modrep-predicate-chain.md sec. 10.

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && not raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_phantom_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s

; CHECK: ScaledModuloReplicationProjection is invalid in the phantom-lane regime
; CHECK-SAME: max_flat_workgroup_size=32 < target wave size 64
; CHECK-NOT: define amdgpu_kernel void @scaled_modrep_phantom_refuse_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	scaled_modrep_phantom_refuse_kernel
	.type	scaled_modrep_phantom_refuse_kernel,@function
scaled_modrep_phantom_refuse_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	v_mov_b32_e32 v1, 1
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scaled_modrep_phantom_refuse_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
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
    .max_flat_workgroup_size:  32
    .name:                     scaled_modrep_phantom_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         scaled_modrep_phantom_refuse_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
