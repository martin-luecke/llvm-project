; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=divscale_vcc_hi_kernel 2>&1 | %FileCheck %s
;
; On wave32, vcc_hi is a scratch scalar; the ISA does not allow it as a
; div-scale flag destination. Refused rather than silently guessing a lowering.

; CHECK: kernel 'divscale_vcc_hi_kernel'
; CHECK-SAME: v_div_scale_f32 [VALU]
; CHECK-SAME: v_div_scale flag destination is wave32 vcc_hi/exec_hi scratch

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	divscale_vcc_hi_kernel
	.p2align	8
	.type	divscale_vcc_hi_kernel,@function
divscale_vcc_hi_kernel:
	v_div_scale_f32 v5, vcc_hi, v0, v1, v0
	ds_store_b32 v7, v5
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel divscale_vcc_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
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
    .name:           divscale_vcc_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         divscale_vcc_hi_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
