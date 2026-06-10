; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --emit-ir=v_cvt_scalef32_sr_pk8_fp8_f32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --write-hsaco=%t.out --kernel=v_cvt_scalef32_sr_pk8_fp8_f32_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PIPE
; RUN: %llvm-objdump -d %t.out | %FileCheck %s --check-prefix=DISASM
;
; Lift test for the gfx1250 packed-8 scaled FP8 conversion with stochastic
; rounding (`v_cvt_scalef32_sr_pk8_fp8_f32`). Profile VOP_V2I32_V8F32_I32_F32:
;   dst <2 x i32>, src0 <8 x f32>, src1 i32 (SR seed), src2 f32 (scale).
; Same-target gfx1250 -> gfx1250: the raiser emits the native intrinsic and the
; backend re-selects the identical instruction (round-trip identity).

; IR-LABEL: define amdgpu_kernel void @v_cvt_scalef32_sr_pk8_fp8_f32_kernel(
; IR: call <2 x i32> @llvm.amdgcn.cvt.scalef32.sr.pk8.fp8.f32(<8 x float> {{[^,]+}}, i32 {{[^,]+}}, float {{[^)]+}})
; IR-NOT: unsupported instruction

; PIPE: raise_cli: wrote
; PIPE-SAME: v_cvt_scalef32_sr_pk8_fp8_f32_kernel

; DISASM: v_cvt_scalef32_sr_pk8_fp8_f32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cvt_scalef32_sr_pk8_fp8_f32_kernel
	.p2align	8
	.type	v_cvt_scalef32_sr_pk8_fp8_f32_kernel,@function
v_cvt_scalef32_sr_pk8_fp8_f32_kernel:
	s_mov_b32 s2, 0x3f800000
	v_mov_b32_e32 v8, 0x3f800000
	v_mov_b32_e32 v9, 0x40000000
	v_mov_b32_e32 v10, 0x40400000
	v_mov_b32_e32 v11, 0x40800000
	v_mov_b32_e32 v12, 0x40a00000
	v_mov_b32_e32 v13, 0x40c00000
	v_mov_b32_e32 v14, 0x40e00000
	v_mov_b32_e32 v15, 0x41000000
	v_mov_b32_e32 v16, 0
	v_mov_b32_e32 v0, 0
	;;#ASMSTART
	v_cvt_scalef32_sr_pk8_fp8_f32 v[24:25], v[8:15], v16, s2
	;;#ASMEND
	global_store_b64 v0, v[24:25], s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cvt_scalef32_sr_pk8_fp8_f32_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 26
		.amdhsa_next_free_sgpr 3
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
    .name: v_cvt_scalef32_sr_pk8_fp8_f32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 3
    .symbol: v_cvt_scalef32_sr_pk8_fp8_f32_kernel.kd
    .vgpr_count: 26
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
