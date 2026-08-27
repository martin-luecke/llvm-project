; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir=v_add_sub_nc_u32_clamp_kernel 2>/dev/null | %FileCheck %s

; The e64 clamp bit on v_{add,sub,subrev}_nc_u32 requests unsigned saturation;
; it must lift to uadd.sat/usub.sat. Without the bit the ops stay plain
; add/sub. Guards the clamp handling in handleVALU (handle-valu.cpp).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	v_add_sub_nc_u32_clamp_kernel
	.p2align	8
	.type	v_add_sub_nc_u32_clamp_kernel,@function
v_add_sub_nc_u32_clamp_kernel:
; CHECK-LABEL: define amdgpu_kernel void @v_add_sub_nc_u32_clamp_kernel(
; The second source is the inline constant 1 throughout, so the checks also
; pin the operand order that distinguishes sub from subrev.
	v_add_nc_u32 v2, v0, 1 clamp
; CHECK: call i32 @llvm.uadd.sat.i32(i32 %{{.+}}, i32 1)
	v_sub_nc_u32 v3, v0, 1 clamp
; CHECK: call i32 @llvm.usub.sat.i32(i32 %{{.+}}, i32 1)
	v_subrev_nc_u32 v4, v0, 1 clamp
; CHECK: call i32 @llvm.usub.sat.i32(i32 1, i32 %{{.+}})
	v_add_nc_u32_e64 v5, v0, 1
; CHECK: add i32 %{{.+}}, 1
	v_sub_nc_u32_e64 v6, v0, 1
; CHECK: sub i32 %{{.+}}, 1
	v_subrev_nc_u32_e64 v7, v0, 1
; CHECK: sub i32 1, %{{.+}}
; The e32 forms have no clamp operand at all.
	v_add_nc_u32_e32 v8, 1, v0
; CHECK: add i32 1, %{{.+}}
	v_sub_nc_u32_e32 v9, 1, v0
; CHECK: sub i32 1, %{{.+}}
	v_subrev_nc_u32_e32 v10, 1, v0
; CHECK: sub i32 %{{.+}}, 1
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_add_sub_nc_u32_clamp_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 11
		.amdhsa_next_free_sgpr 0
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
    .name:           v_add_sub_nc_u32_clamp_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         v_add_sub_nc_u32_clamp_kernel.kd
    .vgpr_count:     11
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
