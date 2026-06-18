; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1151 \
; RUN:     --emit-ir=cov_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Coverage for VALU ops that gfx1151 (RDNA3.5) supports natively and that map
; 1:1 from gfx1250 (cross-checked against instruction_manual_1250.txt):
;   v_mad_u16            D.u16 = S0.u16*S1.u16 + S2.u16      (95118)
;   v_rsq_f64           1.0/sqrt(S0.f64)                     (77887)
;   v_rndne_f64         round-to-nearest-even f64            (77574)
;   v_fract_f64         S0 + -floor(S0), clamped <1.0        (75548)
;   v_frexp_mant_f32/64 frexp significand                    (71569/72)
;   v_frexp_exp_i32_f32/64 frexp exponent as i32             (75681/75739)
;
; CHECK-LABEL: define amdgpu_kernel void @cov_kernel(
; CHECK-DAG: mul i16
; CHECK-DAG: call double @llvm.amdgcn.rsq.f64(
; CHECK-DAG: call double @llvm.roundeven.f64(
; CHECK-DAG: call double @llvm.amdgcn.fract.f64(
; CHECK-DAG: call float @llvm.amdgcn.frexp.mant.f32(
; CHECK-DAG: call i32 @llvm.amdgcn.frexp.exp.i32.f32(
; CHECK-DAG: call double @llvm.amdgcn.frexp.mant.f64(
; CHECK-DAG: call i32 @llvm.amdgcn.frexp.exp.i32.f64(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cov_kernel
	.p2align	8
	.type	cov_kernel,@function
cov_kernel:
; %bb.0:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	; derive non-constant operands from the workitem id (v0) so no op folds
	v_cvt_f64_u32_e32 v[6:7], v0
	v_mov_b32_e32 v1, v0
	v_mad_u16 v2, v1, v1, v1
	v_rsq_f64_e32 v[8:9], v[6:7]
	v_rndne_f64_e32 v[10:11], v[6:7]
	v_fract_f64_e32 v[12:13], v[6:7]
	v_frexp_mant_f32_e32 v3, v6
	v_frexp_exp_i32_f32_e32 v4, v6
	v_frexp_mant_f64_e32 v[14:15], v[6:7]
	v_frexp_exp_i32_f64_e32 v5, v[6:7]
	v_add_nc_u32_e32 v2, v2, v3
	v_add_nc_u32_e32 v2, v2, v4
	v_add_nc_u32_e32 v2, v2, v5
	v_add_nc_u32_e32 v2, v2, v8
	v_add_nc_u32_e32 v2, v2, v10
	v_add_nc_u32_e32 v2, v2, v12
	v_add_nc_u32_e32 v2, v2, v14
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cov_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 16
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
    .name: cov_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: cov_kernel.kd
    .vgpr_count: 16
    .wavefront_size: 32
amdhsa.version:
  - 1
  - 2
...
	.end_amdgpu_metadata
