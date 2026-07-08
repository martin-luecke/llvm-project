; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --write-hsaco=%t.gfx950.hsaco \
; RUN:        --target-isa=gfx950 2>/dev/null \
; RUN:   && %llvm-objdump -d %t.gfx950.hsaco | %FileCheck %s --check-prefix=GFX950
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --write-hsaco=%t.gfx1250.hsaco \
; RUN:        --target-isa=gfx1250 2>/dev/null \
; RUN:   && %llvm-objdump -d %t.gfx1250.hsaco | %FileCheck %s --check-prefix=SAME

; GFX950-LABEL: <wave_mode_replay_setreg>:
; GFX950-NOT: s_setreg_imm32_b32
; GFX950: s_cbranch_execz
; SAME-LABEL: <wave_mode_replay_setreg>:
; SAME: s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wave_mode_replay_setreg
	.p2align	8
	.type	wave_mode_replay_setreg,@function
wave_mode_replay_setreg:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wave_mode_replay_setreg
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 1
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
    .max_flat_workgroup_size: 256
    .name:           wave_mode_replay_setreg
    .private_segment_fixed_size: 0
    .sgpr_count:     1
    .symbol:         wave_mode_replay_setreg.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
