; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_fp4_fp4_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX942
;
; Cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4 with
; FP4 (E2M1) fragments on BOTH sides. Pins the FP4 -> FP8 (E4M3)
; widening branch of `emitWMMAScaleF8F6F4toMFMA` in `wmma_lowering.cpp`.
;
; FP4 widening is branchless bit-arithmetic (`widenF4NibbleToFP8`):
; sign << 7 | exp << 3 | mant << 2, with a select for the single
; subnormal value (E=00, M=1 -> ±0.5 -> biased FP8 exp = 6, mant = 0)
; and an override for ±0. No LUT, no memory access; per element ~10 IR
; ops, all ALU. After widening, the fragment is 16 fp8 dwords / wave32
; lane (same shape as a native fp8 fragment), so the downstream
; redistribute / MFMA / scale / fadd pipeline is unchanged.
;
; INVARIANTS PINNED:
;
;   1. The widening emits no `load` / no constant-pool ref / no LUT-
;      shaped GEP -- guards the "GPU shouldn't LUT" property.
;
;   2. Per-nibble widening core: an `and ..., 1` (mantissa) combined
;      with `shl ..., 7` (sign placement) and `select i1 ..., i32 6,
;      i32 ...` (subnormal vs normal exp). One of each per nibble; with
;      64 nibbles per lane and 2 matrices, the widening is heavily
;      replicated.
;
;   3. After widening, the K-loop dispatches to the fp8.fp8 MFMA
;      family (both sides widen FP4 -> FP8 E4M3 by mantissa-width match).
;
;   4. Under WaveNative (default), the K-loop runs twice for 8 total
;      bf8.fp8 MFMA calls -- same structure as the f8/bf8 fixture.

; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_fp4_fp4_kernel(

; The widening core: subnormal-vs-normal select on the exp produces
; `select i1 %{{[^,]+}}, i32 6, i32 %{{[^,]+}}` for each nibble. Pin
; one occurrence (the pattern is heavily replicated; checking
; presence is enough).
; IR_GFX942-DAG: select i1 %{{[^,]+}}, i32 6, i32

; Mantissa pad: `shl i32 %{{[^,]+}}, 2` places the single FP4 mantissa
; bit into FP8 bit 2.
; IR_GFX942-DAG: shl i32 %{{[^,]+}}, 2

; FP4 -> fp8.fp8 MFMA dispatch. Both sides widen to E4M3.
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; 8 K-block MFMAs total under WaveNative (4 per pass * 2 passes).
; IR_GFX942-COUNT-7: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; Scale-on-output fmuladd (same as the f8/bf8 path).
; IR_GFX942-DAG: call <4 x float> @llvm.fmuladd.v4f32(

; Negative: no LUT / constant-pool load for the per-element widening.
; A constant-pool reference would surface as `@__const.` or a `load`
; with an attribute that names the constant array; with bit-arithmetic
; only, no such symbol appears.
; IR_GFX942-NOT: @__const.
; IR_GFX942-NOT: load i8

; Negative: no other MFMA combinations.
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8

; Negative: no cross-target dispatch to gfx950 / gfx1250 paths.
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale_f32_16x16x128_fp4_fp4_kernel
	.p2align	8
	.type	wmma_scale_f32_16x16x128_fp4_fp4_kernel,@function
wmma_scale_f32_16x16x128_fp4_fp4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v40, 0
	s_wait_kmcnt 0x0
	s_load_b256 s[0:7], s[36:37], 0x0
	s_load_b256 s[8:15], s[38:39], 0x0
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
	v_mov_b64_e32 v[16:17], s[44:45]
	v_mov_b64_e32 v[18:19], s[46:47]
	v_mov_b64_e32 v[20:21], s[48:49]
	v_mov_b64_e32 v[22:23], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_scale_f32_16x16x128_f8f6f4 v[16:23], v[0:7], v[8:15], v[16:23], s42, s43 matrix_a_fmt:MATRIX_FMT_FP4 matrix_b_fmt:MATRIX_FMT_FP4
	s_clause 0x1
	global_store_b128 v40, v[20:23], s[40:41] offset:16
	global_store_b128 v40, v[16:19], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale_f32_16x16x128_fp4_fp4_kernel
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
    .name:           wmma_scale_f32_16x16x128_fp4_fp4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale_f32_16x16x128_fp4_fp4_kernel.kd
    .vgpr_count:     41
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
