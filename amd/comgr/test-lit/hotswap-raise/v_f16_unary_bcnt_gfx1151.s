; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1151 \
; RUN:     --emit-ir=f16u_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; f16 unary special-functions / rounding + v_bcnt_u32_b32. gfx1151 (RDNA3.5)
; has the native f16 set; each maps to the matching hardware intrinsic on f16.
; v_bcnt_u32_b32 = popcount(S0) + S1 (manual: count 1-bits, accumulate S1).
;
; CHECK-LABEL: define amdgpu_kernel void @f16u_kernel(
; CHECK-DAG: call half @llvm.amdgcn.rcp.f16(
; CHECK-DAG: call half @llvm.amdgcn.rsq.f16(
; CHECK-DAG: call half @llvm.amdgcn.sqrt.f16(
; CHECK-DAG: call half @llvm.amdgcn.sin.f16(
; CHECK-DAG: call half @llvm.amdgcn.cos.f16(
; CHECK-DAG: call half @llvm.roundeven.f16(
; CHECK-DAG: call half @llvm.amdgcn.fract.f16(
; CHECK-DAG: call i32 @llvm.ctpop.i32(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	f16u_kernel
	.p2align	8
	.type	f16u_kernel,@function
f16u_kernel:
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_cvt_f16_u16_e32 v1, v0
	v_rcp_f16_e32 v2, v1
	v_rsq_f16_e32 v3, v1
	v_sqrt_f16_e32 v4, v1
	v_sin_f16_e32 v5, v1
	v_cos_f16_e32 v6, v1
	v_rndne_f16_e32 v7, v1
	v_fract_f16_e32 v8, v1
	v_bcnt_u32_b32_e64 v9, v0, v0
	v_add_nc_u32_e32 v2, v2, v9
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel f16u_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 10
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
    .name: f16u_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: f16u_kernel.kd
    .vgpr_count: 10
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
