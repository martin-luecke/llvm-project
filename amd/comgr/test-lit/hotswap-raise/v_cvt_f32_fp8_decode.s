; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS \
; RUN:       --implicit-check-not=llvm.amdgcn.cvt.pk.f32 \
; RUN:       --implicit-check-not=llvm.amdgcn.cvt.f32.fp8 \
; RUN:       --implicit-check-not=_dec_fnuz

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME --implicit-check-not=_dec_

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_dec_kernel
	.p2align	8
	.type	cvt_dec_kernel,@function
cvt_dec_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x40404040
; Cross-target: decode the source-format byte exactly in IR rather than
; re-encoding it into the target's narrower format first. 5 fp8 bytes here
; (packed pair, single, op_sel:[1] pair) plus a bf8 pair; bf8 is E5M2, the one
; with Inf, which only survives on this path. The RUN line asserts no decode
; intrinsic and no FNUZ-direction decode appear anywhere.
; CROSS-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; CROSS-COUNT-5: %fp8_dec_ocp{{[0-9]*}} = select
; CROSS-DAG: %bf8_dec_ocp{{[0-9]*}} = select
; byte_sel picks the lane. gfx1250 prints it as `byte_sel:N`, never `op_sel:`,
; so a textual op_sel guard never sees it and byte 1/2/3 would collapse to 0.
; CROSS-DAG: lshr i32 %{{[^,]+}}, 16
; Same-format target keeps the hardware decode, op_sel:[1] included, and
; builds no software decode at all.
; SAME-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 false)
; SAME: call float @llvm.amdgcn.cvt.f32.fp8(
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 true)
; SAME: call float @llvm.amdgcn.cvt.f32.fp8(i32 %{{[^,]+}}, i32 2)
	v_cvt_pk_f32_fp8 v[2:3], v0
	v_cvt_f32_fp8 v4, v0
	v_cvt_pk_f32_fp8 v[6:7], v0 op_sel:[1,0]
	v_cvt_pk_f32_bf8 v[10:11], v0
	v_cvt_f32_fp8 v12, v0 byte_sel:2
	v_mov_b32_e32 v8, 0
	s_wait_kmcnt 0x0
	global_store_b96 v8, v[2:4], s[0:1]
	global_store_b64 v8, v[6:7], s[0:1] offset:16
	global_store_b64 v8, v[10:11], s[0:1] offset:24
	global_store_b32 v8, v12, s[0:1] offset:32
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_dec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 13
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
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
    .name:           cvt_dec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         cvt_dec_kernel.kd
    .vgpr_count:     13
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
