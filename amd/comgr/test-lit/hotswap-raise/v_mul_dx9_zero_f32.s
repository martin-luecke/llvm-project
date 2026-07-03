; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_mul_dx9_zero_f32_kernel 2>&1 | %FileCheck %s --check-prefix=IR
; RUN: raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=v_mul_dx9_zero_f32_kernel 2>&1 | %FileCheck %s --check-prefix=SAME
;
; v_mul_dx9_zero_f32 is the gfx11+ asm mnemonic for the V_MUL_LEGACY_F32
; pseudo: a DX9-style multiply where 0.0 * x -> +0.0 for any x (including
; NaN/Inf). It must lift to llvm.amdgcn.fmul.legacy -- NOT a plain fmul -- so
; that re-lowering on the target selects the matching legacy multiply and the
; zero-flush semantics are preserved across the gfx1250 -> gfx942 transpile.

; IR-LABEL: define amdgpu_kernel void @v_mul_dx9_zero_f32_kernel(
; IR: call float @llvm.amdgcn.fmul.legacy(
; IR-NOT: fmul float
; IR-NOT: UnsupportedOpcode
; IR: ret void

; SAME-LABEL: define amdgpu_kernel void @v_mul_dx9_zero_f32_kernel(
; SAME: call float @llvm.amdgcn.fmul.legacy(
; SAME-NOT: UnsupportedOpcode

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_mul_dx9_zero_f32_kernel
	.p2align	8
	.type	v_mul_dx9_zero_f32_kernel,@function
v_mul_dx9_zero_f32_kernel:
	v_mul_dx9_zero_f32 v0, v0, v1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_mul_dx9_zero_f32_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_mul_dx9_zero_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mul_dx9_zero_f32_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
