; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_inv_kernel 2>/dev/null | %FileCheck %s
;
; Lift fixture for the gfx12+ GLOBAL cache-control op `global_inv`.
; The handler lifts to an LLVM `fence acquire` with a syncscope derived
; from the CPol scope bits; SIMemoryLegalizer re-emits the appropriate
; target cache control. The lifted IR is target-independent, so a single
; --target-isa run covers it.

; CHECK-LABEL: define amdgpu_kernel void @global_inv_kernel(
; CHECK: fence syncscope("agent") acquire
; CHECK-NEXT: fence syncscope("cluster") acquire
; CHECK-NEXT: fence syncscope("workgroup") acquire
; CHECK-NEXT: fence acquire
; CHECK-NOT: fence

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_inv_kernel
	.p2align	8
	.type	global_inv_kernel,@function
global_inv_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	global_inv scope:SCOPE_DEV
	global_inv scope:SCOPE_SE
	global_inv scope:SCOPE_CU
	global_inv scope:SCOPE_SYS
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_inv_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 0
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
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           global_inv_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         global_inv_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
