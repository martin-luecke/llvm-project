; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --emit-ir=permlane_var_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --write-hsaco=%t.out --kernel=permlane_var_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PIPE
; RUN: %llvm-objdump -d %t.out | %FileCheck %s --check-prefix=DISASM
;
; Lift test for the gfx1250 variable-selector permute
; (`v_permlane16_var_b32` / `v_permlanex16_var_b32`). The per-lane VGPR
; selector (src1) distinguishes these from the immediate-selector
; v_permlane16_b32 family. Same-target gfx1250 -> gfx1250 emits the native
; intrinsic and the backend re-selects the identical instruction.

; IR-LABEL: define amdgpu_kernel void @permlane_var_kernel(
; IR: call i32 @llvm.amdgcn.permlane16.var(i32 {{[^,]+}}, i32 {{[^,]+}}, i32 {{[^,]+}}, i1 {{[^,]+}}, i1 {{[^)]+}})
; IR: call i32 @llvm.amdgcn.permlanex16.var(i32 {{[^,]+}}, i32 {{[^,]+}}, i32 {{[^,]+}}, i1 {{[^,]+}}, i1 {{[^)]+}})
; IR-NOT: unsupported instruction

; PIPE: raise_cli: wrote
; PIPE-SAME: permlane_var_kernel

; DISASM: v_permlane16_var_b32
; DISASM: v_permlanex16_var_b32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	permlane_var_kernel
	.p2align	8
	.type	permlane_var_kernel,@function
permlane_var_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	v_mov_b32_e32 v2, 2
	v_permlane16_var_b32 v0, v1, v2
	v_permlanex16_var_b32 v3, v1, v2
	global_store_b32 v0, v3, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel permlane_var_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: permlane_var_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: permlane_var_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
