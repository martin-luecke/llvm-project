; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_cvt_pkrtz_f16_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; v_cvt_pkrtz_f16_f32 uses target-independent explicitly rounded truncations.
; CHECK-LABEL: define amdgpu_kernel void @v_cvt_pkrtz_f16_f32_kernel(
; CHECK-DAG: [[LO:%pkrtz_lo[0-9]*]] = call half @llvm.fptrunc.round.f16.f32(float %{{[^,]+}}, metadata !"round.towardzero")
; CHECK-DAG: [[HI:%pkrtz_hi[0-9]*]] = call half @llvm.fptrunc.round.f16.f32(float %{{[^,]+}}, metadata !"round.towardzero")
; CHECK: insertelement <2 x half> poison, half [[LO]], i32 0
; CHECK: insertelement <2 x half> %{{[^,]+}}, half [[HI]], i32 1
; CHECK-NOT: @llvm.amdgcn.cvt.pkrtz

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_pkrtz_f16_f32_kernel
	.p2align	8
	.type	v_cvt_pkrtz_f16_f32_kernel,@function
v_cvt_pkrtz_f16_f32_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s6
	v_mov_b32_e32 v2, s7
	v_cvt_pk_rtz_f16_f32_e64 v3, v1, v2
	global_store_b32 v0, v3, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_pkrtz_f16_f32_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
      - { .offset:        12, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_cvt_pkrtz_f16_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_cvt_pkrtz_f16_f32_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
