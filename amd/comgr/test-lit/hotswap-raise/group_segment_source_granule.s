; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=group_segment_source_granule 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR

; A non-zero source LDS request allocates at least one source-subtarget
; granule. Preserve that allocation when the target has a smaller granule.
; gfx1250's authoritative LDS granule is 2048 bytes.
; IR-LABEL: define amdgpu_kernel void @group_segment_source_granule(
; IR: store i32 {{.*}}, ptr addrspace(3)
; IR: attributes #{{[0-9]+}} = { {{.*}}"amdgpu-lds-size"="2048,2048"{{.*}} }

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl group_segment_source_granule
	.p2align 8
	.type group_segment_source_granule,@function
group_segment_source_granule:
	ds_store_b32 v0, v0 offset:1020
	s_wait_dscnt 0
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel group_segment_source_granule
		.amdhsa_group_segment_fixed_size 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 1
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 32
    .name: group_segment_source_granule
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: group_segment_source_granule.kd
    .vgpr_count: 1
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
