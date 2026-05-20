; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_fp6_fp6_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX942
;
; Cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4 with
; FP6 (E2M3) fragments on BOTH sides. Pins the FP6 -> FP8 (E4M3)
; widening branch of `emitWMMAScaleF8F6F4toMFMA`.
;
; FP6 widening is branchless bit-arithmetic
; (`widenFP6NibbleToFP8` + `extractF6Nibble`):
;   * Per-element extract: 6-bit elements packed contiguously across
;     byte / dword boundaries (12 dwords / 64 elements / lane).
;     53 of 64 elements fit in one dword (`shr`+`and`); 11 straddle a
;     dword boundary and need a 2-dword combine (`zext`+`shl`+`or`
;     +`lshr`+`trunc`+`and`).
;   * Per-element widen: `@llvm.ctlz.i32` on the 3-bit mantissa for
;     the subnormal renormalization, `select` for subnormal-vs-normal
;     paths, `select` for the +-0 override. No LUT, no memory access.
;
; INVARIANTS PINNED:
;
;   1. Bit-unpack uses `lshr i64` for the cross-dword elements (the
;     11 boundary cases produce a 64-bit combine + shift). Pin one.
;
;   2. Per-element widening core: `@llvm.ctlz.i32` for the subnormal
;     mantissa leading-zero count, and the `sub i32 6, %{{...}}` /
;     `sub i32 %{{...}}, 29` arithmetic that converts to the
;     destination biased exponent.
;
;   3. FP6 -> fp8.fp8 MFMA dispatch. Both sides widen to E4M3.
;
;   4. Under WaveNative (default), 8 fp8.fp8 MFMA calls (4 K-blocks x
;     2 source-wave passes), same structure as the f8/bf8 path.

; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_fp6_fp6_kernel(

; Cross-dword bit unpack: i64 shift right by a constant amount in [0,32)
; surfaces for the 11 boundary elements.
; IR_GFX942-DAG: lshr i64

; Subnormal renormalization via ctlz on the 3-bit mantissa, vectorized
; across 16 elements per super-chunk (`<16 x i32>` lane vector).
; IR_GFX942-DAG: call <16 x i32> @llvm.ctlz.v16i32(

; Subnormal biased exponent: `6 - lz` from ctlz output (ctlz returns
; 29 for the smallest mantissa = 4, etc; we subtract 29 to get lz in
; [0..2]). Vectorized: `sub <16 x i32> ..., splat (i32 29)`.
; IR_GFX942-DAG: sub <16 x i32> %{{[^,]+}}, {{(splat \(i32 29\)|<i32 29)}}

; FP6 -> fp8.fp8 MFMA dispatch.
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; 8 K-block MFMAs total under WaveNative.
; IR_GFX942-COUNT-7: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; Scale-on-output fmuladd.
; IR_GFX942-DAG: call <4 x float> @llvm.fmuladd.v4f32(

; Negative: no LUT.
; IR_GFX942-NOT: @__const.

; Negative: no other MFMA combinations.
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8

; Negative: no native scaled-WMMA or scaled-MFMA dispatch.
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale_f32_16x16x128_fp6_fp6_kernel
	.p2align	8
	.type	wmma_scale_f32_16x16x128_fp6_fp6_kernel,@function
wmma_scale_f32_16x16x128_fp6_fp6_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v40, 0
	s_wait_kmcnt 0x0
	s_load_b256 s[0:7], s[36:37], 0x0
	s_load_b128 s[8:11], s[36:37], 0x20
	s_load_b256 s[12:19], s[38:39], 0x0
	s_load_b128 s[20:23], s[38:39], 0x20
	s_load_b256 s[44:51], s[40:41], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	v_mov_b64_e32 v[24:25], s[44:45]
	v_mov_b64_e32 v[26:27], s[46:47]
	v_mov_b64_e32 v[28:29], s[48:49]
	v_mov_b64_e32 v[30:31], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[0:11], v[12:23], v[24:31], s42, s43 matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP6
	s_clause 0x1
	global_store_b128 v40, v[28:31], s[40:41] offset:16
	global_store_b128 v40, v[24:27], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale_f32_16x16x128_fp6_fp6_kernel
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 41
		.amdhsa_next_free_sgpr 52
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
      - { .offset:         24, .size:           4, .value_kind:     by_value }
      - { .offset:         28, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           wmma_scale_f32_16x16x128_fp6_fp6_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale_f32_16x16x128_fp6_fp6_kernel.kd
    .vgpr_count:     41
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
