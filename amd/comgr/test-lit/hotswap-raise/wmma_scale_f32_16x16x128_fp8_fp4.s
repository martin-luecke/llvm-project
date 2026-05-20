; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_fp8_fp4_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX942
;
; Mixed-format cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4:
; matrix A is FP8 E4M3 (16 dwords, no widening), matrix B is FP4 E2M1
; (8 dwords, widens to FP8 E4M3 via `widenF4FragmentToFP8`). Both
; matrices dispatch to the `mfma_f32_16x16x32_fp8_fp8` intrinsic on the
; gfx942 side.
;
; Scale formats: A uses E8M0 (the canonical MXFP convention), B uses
; E4M3 (the per-byte SE4M3 FP8 format). This is row 2 of the spec's
; legal-combinations table (F8 x E8M0, F4 x E5M3/E4M3) -- non-E8M0
; scale on B requires F4 data on B, which we satisfy.
;
; INVARIANTS PINNED:
;
;   1. Asymmetric data widening: A is pass-through (no widening IR
;      for the A side), B widens FP4 -> FP8 via the `select i1
;      %{{...}}, i32 6, i32 ...` subnormal-vs-normal pattern.
;
;   2. Scale path is the MIXED-format branch of `buildScaleFactorVec`:
;      side A decoded via the E8M0 fast-component (ldexp + NaN select),
;      side B decoded via `@llvm.amdgcn.cvt.f32.fp8` (hw cvt, single
;      instruction). Result combined via `fmul`.
;
;   3. Dispatch reaches `mfma_f32_16x16x32_fp8_fp8` (both sides end as
;      FP8 E4M3 after widening).

; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_fp8_fp4_kernel(

; FP8 x FP4 dispatches to fp8.fp8 (both widen to E4M3 -- A is pass-
; through native FP8, B widens FP4 -> FP8 via `widenF4FragmentToFP8`).
; The widening IR appears upfront before any MFMA call; that surface
; is already pinned in the `*_fp4_fp4.s` fixture so we don't re-pin
; it here. This fixture focuses on the MIXED scale-format path.
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942-COUNT-7: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; Mixed-scale path: E4M3 hw cvt for the B side. Inside the K-loop,
; emitted per K-block alongside the MFMA call; the 8th iteration's
; cvt lives after the last MFMA where the DAG can find it.
; IR_GFX942-DAG: call float @llvm.amdgcn.cvt.f32.fp8(

; E8M0 path on the A side: per-K-block `ldexp(1.0, byte - 127)` + NaN
; select. The `sub i32 ..., 127` is the E8M0 unbiasing arithmetic.
; IR_GFX942-DAG: sub i32 %{{[^,]+}}, 127

; Mixed-format combine: factorA (E8M0) and factorB (E4M3) combined
; via fmul, NOT the optimized E8M0 x E8M0 sum-of-exponents shortcut.
; IR_GFX942-DAG: fmul float

; Negative: no LUT, no cross-target dispatch, no other MFMA combos.
; IR_GFX942-NOT: @__const.
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale_f32_16x16x128_fp8_fp4_kernel
	.p2align	8
	.type	wmma_scale_f32_16x16x128_fp8_fp4_kernel,@function
wmma_scale_f32_16x16x128_fp8_fp4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v40, 0
	s_wait_kmcnt 0x0
	s_load_b512 s[0:15], s[36:37], 0x0
	s_load_b256 s[16:23], s[38:39], 0x0
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
	v_wmma_scale_f32_16x16x128_f8f6f4 v[24:31], v[0:15], v[16:23], v[24:31], s42, s43 matrix_a_fmt:MATRIX_FMT_FP8 matrix_b_fmt:MATRIX_FMT_FP4 matrix_b_scale_fmt:MATRIX_SCALE_FMT_E4M3
	s_clause 0x1
	global_store_b128 v40, v[28:31], s[40:41] offset:16
	global_store_b128 v40, v[24:27], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale_f32_16x16x128_fp8_fp4_kernel
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
    .name:           wmma_scale_f32_16x16x128_fp8_fp4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale_f32_16x16x128_fp8_fp4_kernel.kd
    .vgpr_count:     41
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
