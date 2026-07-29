	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6

	.text
	.globl first_kernel
	.p2align 8
	.type first_kernel,@function
first_kernel:
	s_endpgm

	.globl second_kernel
	.p2align 8
	.type second_kernel,@function
second_kernel:
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6, 0x0
	.amdhsa_kernel first_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel
	.p2align 6, 0x0
	.amdhsa_kernel second_kernel
		.amdhsa_kernarg_size 280
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
	.end_amdhsa_kernel

	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 1024
    .name: first_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: first_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 64
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 4
    .kernarg_segment_size: 280
    .max_flat_workgroup_size: 1024
    .name: second_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: second_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
