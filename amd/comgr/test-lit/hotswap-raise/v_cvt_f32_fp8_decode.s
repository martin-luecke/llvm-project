; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=CROSS

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=cvt_dec_kernel \
; RUN:   | %FileCheck %s --check-prefix=SAME

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	cvt_dec_kernel
	.p2align	8
	.type	cvt_dec_kernel,@function
cvt_dec_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v0, 0x40404040
; Cross-target decode does NOT route through the target's fp8 hardware. byte
; -> f32 is widening, so the source-format byte is decoded exactly in IR
; (%fp8_dec_ocp), which keeps OCP E4M3's (240,448] and -0 instead of clipping
; them to the target's narrower FNUZ range. Five bytes are decoded here: 2 for
; the packed pair, 1 for the single-byte form, 2 for the op_sel:[1] pair.
; CROSS-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; CROSS-COUNT-5: %fp8_dec_ocp{{[0-9]*}} = select
; bf8 is OCP E5M2, which unlike E4M3 has Inf -- the software decode keeps it
; where the old FNUZ round trip turned it into a finite 57344 (and, after the
; Inf->NaN policy fix, into NaN).
; CROSS-DAG: %bf8_dec_ocp{{[0-9]*}} = select
; A use of either decode intrinsic would leave a declare after this point.
; CROSS-NOT: @llvm.amdgcn.cvt.pk.f32.fp8
; CROSS-NOT: @llvm.amdgcn.cvt.f32.fp8
; CROSS-NOT: @llvm.amdgcn.cvt.pk.f32.bf8
; CROSS-NOT: e4m3_fnuz
; Same-format target keeps the hardware decode and never builds one in IR;
; op_sel:[1] still rides on the intrinsic's own word_sel.
; SAME-LABEL: define amdgpu_kernel void @cvt_dec_kernel(
; SAME-NOT: fp8_dec
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 false)
; SAME: call float @llvm.amdgcn.cvt.f32.fp8(
; SAME: call <2 x float> @llvm.amdgcn.cvt.pk.f32.fp8(i32 %{{[^,]+}}, i1 true)
; SAME-NOT: fp8_dec
	v_cvt_pk_f32_fp8 v[2:3], v0
	v_cvt_f32_fp8 v4, v0
	v_cvt_pk_f32_fp8 v[6:7], v0 op_sel:[1,0]
	v_cvt_pk_f32_bf8 v[10:11], v0
	v_mov_b32_e32 v8, 0
	s_wait_kmcnt 0x0
	global_store_b96 v8, v[2:4], s[0:1]
	global_store_b64 v8, v[6:7], s[0:1] offset:16
	global_store_b64 v8, v[10:11], s[0:1] offset:24
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cvt_dec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 12
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
    .vgpr_count:     12
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
