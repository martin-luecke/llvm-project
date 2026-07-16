; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_bcnt_u32_b32_kernel \
; RUN:   | %FileCheck %s

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_bcnt_u32_b32_kernel
	.p2align	8
	.type	v_bcnt_u32_b32_kernel,@function
v_bcnt_u32_b32_kernel:
; CHECK-LABEL: define amdgpu_kernel void @v_bcnt_u32_b32_kernel(
	; Lifts per-lane to popcount(src0) + src1; must not be refused when
	; widening wave32 -> wave64.
	; CHECK: %bcnt_u32{{[0-9]*}} = call i32 @llvm.ctpop.i32(i32 %{{[^)]+}})
	; CHECK: %bcnt_u32_add{{[0-9]*}} = add i32 %bcnt_u32{{[0-9]*}}, 1
	v_bcnt_u32_b32 v1, v0, 1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_bcnt_u32_b32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_bcnt_u32_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_bcnt_u32_b32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
