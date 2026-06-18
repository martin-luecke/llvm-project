; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1151 \
; RUN:     --emit-ir=v_sin_cos_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; gfx1250 VOP1 f32 trig (v_sin_f32 / v_cos_f32) lower 1:1 to gfx11: both ISAs
; use the same revolution-domain hardware function (result = sin/cos(2*pi*src)),
; so the lift emits the amdgcn.sin / amdgcn.cos intrinsics, which the gfx1151
; backend re-lowers to native v_sin_f32 / v_cos_f32. Exercised by RoPE kernels.
;
; CHECK-LABEL: define amdgpu_kernel void @v_sin_cos_f32_kernel(
; CHECK-DAG: call float @llvm.amdgcn.sin.f32(
; CHECK-DAG: call float @llvm.amdgcn.cos.f32(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_sin_cos_f32_kernel
	.p2align	8
	.type	v_sin_cos_f32_kernel,@function
v_sin_cos_f32_kernel:
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, 0x3e800000
	v_sin_f32_e32 v2, v1
	v_cos_f32_e32 v3, v1
	v_add_f32_e32 v2, v2, v3
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_sin_cos_f32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name: v_sin_cos_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: v_sin_cos_f32_kernel.kd
    .vgpr_count: 4
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
