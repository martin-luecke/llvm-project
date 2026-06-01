; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_dual_add_f32_kernel 2>&1 | %FileCheck %s
;
; We use CHECK-DAG to revert the order of the checks, we first look for the fadd then for the input.
; CHECK-LABEL: define amdgpu_kernel void @v_dual_add_f32_kernel
; CHECK-DAG: fadd float %[[VGPR0:.+]], %[[CONST_1:.+]]
; CHECK-DAG: fadd float %[[VGPR1:.+]], %[[CONST_2:.+]]
; CHECK-DAG: %[[CONST_1]] = bitcast i32 1065353216 to float
; CHECK-DAG: %[[CONST_2]] = bitcast i32 1073741824 to float
; CHECK-DAG: %[[VGPR0]] = bitcast i32 %{{.+}} to float
; CHECK-DAG: %[[VGPR1]] = bitcast i32 %{{.+}} to float

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_dual_add_f32_kernel
	.p2align	8
	.type	v_dual_add_f32_kernel,@function
v_dual_add_f32_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s0 
	v_mov_b32_e32 v1, s1
    v_mov_b32_e32 v2, 1.0
    v_mov_b32_e32 v3, 2.0
    v_dual_add_f32 v4, v0, v2 :: v_dual_add_f32 v5, v1, v3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_dual_add_f32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
  - .args:
      - { .address_space:  generic, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           v_dual_add_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         v_dual_add_f32_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
