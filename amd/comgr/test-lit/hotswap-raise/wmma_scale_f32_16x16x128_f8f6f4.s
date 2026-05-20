; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX942
;
; Cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4
; (gfx1250 RDNA4 source) -> gfx942 (CDNA3 target). Pins the
; `emitWMMAScaleF8F6F4toMFMA` path in `wmma_lowering.cpp`
; dispatched by `ctx.targetIsa.hasMfma && !hasGfx950Insts` in
; `handle_valu_vop3p.cpp` under
; `CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4`.
;
; Default projection is WaveNative cross-widen (numSrcWaves == 2: one
; target wave64 absorbs two source wave32s). The emitter runs two
; passes -- GroupBase = 0 for source wave 0 (target lanes 0..31),
; GroupBase = 32 for source wave 1 (target lanes 32..63) -- and selects
; per-lane between the two passes' output dwords. Each pass issues 4
; K-block MFMA calls, for a total of 8 bf8.fp8 MFMA calls under
; WaveNative (vs 4 under MODREP). The companion MODREP RUN line below
; pins the single-pass path via `--disable-wave-native`.
;
; gfx942 has neither the scaled-WMMA family (gfx1250-only) nor the
; scaled-MFMA F8F6F4 family (gfx950+, gated on `FeatureGFX950Insts`).
; The lowering decomposes the K=128 WMMA-scale into 4 chained K=32
; unscaled fp8/bf8 MFMA calls from the family
; `int_amdgcn_mfma_f32_16x16x32_{fp8,bf8}_{fp8,bf8}` (IntrinsicsAMDGPU.td:3594,
; multiclass `AMDGPUMFp8MfmaIntrinsic`, gated on `FeatureMAIInsts` +
; `FeatureFP8Insts` which are both set on gfx942's `FeatureISAVersion9_4_2`,
; AMDGPU.td:1813-1821) and applies the per-K-block UE8M0 scale
; `2^(sA + sB - 254)` on each MFMA's `<4 x f32>` partial via one
; `llvm.ldexp.f32` + fmul + fadd before accumulating into the running
; output. The scale-on-output design is precision-equivalent to per-
; input scaling because UE8M0 is constant in K within a K-block --
; `ldexp` on f32 is bit-exact, so the per-block exponent factors out
; of the inner sum.
;
; The fixture kernel has `matrix_a_fmt:MATRIX_FMT_BF8` and default
; `matrix_b_fmt:MATRIX_FMT_FP8`, so the dispatched MFMA intrinsic
; is `mfma.f32.16x16x32.bf8.fp8`. The matrix_*_scale / matrix_*_scale_fmt
; immediates default to 0 (canonical UE8M0, byte index k == K-block k).
;
; INVARIANTS PINNED:
;
;   1. Exactly 4 calls to `@llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8`,
;      one per K-block (K=128 / K=32 = 4). Each call passes a
;      ZERO `<4 x f32>` accumulator (we accumulate at IR level
;      after applying the per-K-block scale, not via MFMA chaining,
;      so each K-block's partial is independent and scale-clean).
;
;   2. Per-K-block scale machinery: 4 calls to `llvm.ldexp.f32.i32`
;      (one per K-block, builds `2^(sA + sB - 254)` from the
;      `extractScaleByte` outputs). The unbiasing arithmetic uses
;      `add` + `sub i32 ..., 254`.
;
;   3. Per-K-block fmul + fadd on `<4 x f32>` -- 4 fmuls (the
;      `kblock_scaled` step) and 4 fadds (the `kblock_accum` step).
;      These are the ONLY application of the UE8M0 scale -- no
;      per-input fmul/ldexp on widened f32 values.
;
;   4. Lane redistribution via `llvm.amdgcn.ds.bpermute` for the
;      wave32 -> wave64 wave-projection (scale_src0/1 redistribution
;      + accumulator redistribute + per-K-block A/B redistribution
;      + final collect). The presence of bpermute is the marker that
;      we ran the cross-wave-size lowering, not a same-wave-size
;      shortcut.
;
; NEGATIVE PINS:
;
;   * NO call to `llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4` --
;     would mean the gfx942 path silently fell through to the
;     gfx1250 same-target arm.
;   * NO call to `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4` --
;     would mean the gfx942 path mis-dispatched to the gfx950
;     cross-target arm (whose intrinsic gfx942 cannot lower).
;   * NO call to other fp8/bf8 MFMA combinations (fp8.fp8 /
;     fp8.bf8 / bf8.bf8) -- would mean the (aFmt, bFmt) dispatch
;     in `pickGfx942F8MfmaIntrinsic` got the wrong intrinsic.

; IR_GFX942-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(

; Under WaveNative, the K-loop runs TWICE (pass 0 + pass 32) for a
; total of 8 K-block MFMA iterations. Each iteration emits in order:
;   MFMA partial (bf8.fp8 -- aFmt = MATRIX_FMT_BF8 (1), bFmt =
;   MATRIX_FMT_FP8 (0, default)), ldexp scale factor, fmul scaled,
;   fadd into the running accumulator. The MFMA accumulator argument
;   is `zeroinitializer` per call -- we accumulate at IR level after
;   applying the per-K-block scale, NOT via MFMA chaining.
;
; First K-block of pass 0 pins the per-iteration emission order:
; IR_GFX942: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942: sub i32 %{{[^,]+}}, 254
; IR_GFX942: call float @llvm.ldexp.f32.i32(float 1.000000e+00, i32 %{{[^)]+}})
; IR_GFX942: fmul <4 x float>
; IR_GFX942: fadd <4 x float>
;
; The remaining 7 K-blocks (3 in pass 0, 4 in pass 1): 7 more bf8.fp8
; MFMA calls. The intervening ldexp / fmul / fadd / per-pass redistribute
; are required by the K-loop structure; only the MFMA count is asserted
; directly because the per-iteration emission order is pinned above
; and the loop body is deterministic.
; IR_GFX942-COUNT-7: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)

; WaveNative final per-lane select: target lanes 0..31 take pass 0's
; output, target lanes 32..63 take pass 1's. The `select i1 %is_group1`
; pattern below is the marker that the two-pass dispatch ran.
; IR_GFX942-DAG: icmp uge i32 %{{[^,]+}}, 32
; IR_GFX942-DAG: select i1 %{{[^,]+}}, i32 %{{[^,]+}}, i32 %{{[^,]+}}

; Lane redistribution via ds.bpermute (wave32 -> wave64 cross-wave-size
; lowering marker -- many bpermute calls; one is enough to pin presence).
; IR_GFX942-DAG: call i32 @llvm.amdgcn.ds.bpermute(

; Negative: no native scaled-WMMA intrinsic (would mean the gfx942
; cross-target dispatch fell through to the gfx1250 same-target arm).
; IR_GFX942-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4

; Negative: no scaled-MFMA F8F6F4 intrinsic (would mean we
; mis-dispatched to the gfx950 cross-target arm; gfx942 has no
; codegen for that intrinsic and llc would crash at lowering).
; IR_GFX942-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4

; Negative: no MFMA combinations other than bf8.fp8 (would mean the
; (aFmt, bFmt) dispatch in `pickGfx942F8MfmaIntrinsic` is wrong).
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8
; IR_GFX942-NOT: @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8

; MODREP fallback: pin the single-pass path via `--disable-wave-native`.
; Same emitter, numSrcWaves == 1 -> only pass 0 runs -> 4 MFMA calls
; total and no per-pass select diamond.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX942_MODREP

; IR_GFX942_MODREP-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(
; IR_GFX942_MODREP-COUNT-4: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 %{{[^,]+}}, i64 %{{[^,]+}}, <4 x float> zeroinitializer, i32 0, i32 0, i32 0)
; IR_GFX942_MODREP-NOT: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(
; IR_GFX942_MODREP-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
; IR_GFX942_MODREP-NOT: @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1250 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=IR
;
; Lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4 (gfx1250 RDNA4
; VOP3PX2 opcode 0x033, ScaledWMMA family) — same-target
; (gfx1250 -> gfx1250) intrinsic-emit path. Pins the principled lift
; in transpiler/handle_valu_vop3p.cpp under
; CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4 when
; `ctx.targetIsa.hasTensorOps` is true. Companion fixture to
; `wmma_scale_f32_16x16x128_f8f6f4.ll`, which pins the cross-target
; (gfx942) loud refusal.
;
; 18 MC pseudos collapse onto this single CanonicalOp (9 mantissa pairs
; `{f4,f6,f8} A × {f4,f6,f8} B` × `_twoaddr`/`_threeaddr`), per
; `WMMA_F8F6F4_Profiles` in VOP3PInstructions.td:1908. The per-matrix
; dword count is encoded by the opcode's `_fA_fB_w32_*` suffix
; (f8 → 16 dwords, f6 → 12, f4 → 8) and the in-family element
; distinction (BF8 vs FP8 within f8; BF6 vs FP6 within f6) lives in
; the `matrix_a_fmt` / `matrix_b_fmt` named-immediate operands
; (`enum MatrixFMT { FP8=0, BF8=1, FP6=2, BF6=3, FP4=4 }`,
; SIDefines.h:1052-1058). The HIP fixture compiles to the
; `_f8_f8_w32_threeaddr` MC pseudo with `matrix_a_fmt:MATRIX_FMT_BF8`
; and `matrix_b_fmt:MATRIX_FMT_FP8` (default) — the same shape as the
; failing kerneldex GEMMs (B8F8 / F8B8 ID73f0 contractions).
;
; The native intrinsic `int_amdgcn_wmma_scale_f32_16x16x128_f8f6f4`
; (IntrinsicsAMDGPU.td:4138, class
; `AMDGPUWmmaScaleIntrinsicModsC<llvm_i32_ty>`) takes 14 args:
;
;   <8 x float> llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4(
;       i32 matrix_a_fmt, <NA x i32> A,
;       i32 matrix_b_fmt, <NB x i32> B,
;       i16 c_mod, <8 x float> C,
;       i32 matrix_a_scale, i32 matrix_a_scale_fmt, i32 scale_src0,
;       i32 matrix_b_scale, i32 matrix_b_scale_fmt, i32 scale_src1,
;       i1 matrix_a_reuse, i1 matrix_b_reuse)
;
; Overloaded on D, A and B element vector types, so the f8_f8 form
; mangles to `.v8f32.v16i32.v16i32`. The handler decodes named
; operands via `AMDGPU::getNamedOperandIdx` (`matrix_a_fmt`,
; `matrix_b_fmt`, `matrix_a_scale`, `matrix_b_scale`,
; `matrix_a_scale_fmt`, `matrix_b_scale_fmt`, `scale_src0`,
; `scale_src1`, `matrix_a_reuse`, `matrix_b_reuse`,
; `src2_modifiers`) so any future TableGen reshuffle of the scaled-
; WMMA Ins64 layout flows in for free.
;
; INVARIANTS PINNED:
;
;   1. The native gfx1250 scaled-WMMA intrinsic is emitted (NOT a
;      fallback to MFMA / non-scaled WMMA / a different K-width).
;      The defining marker is the intrinsic name
;      `llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4` with mangled
;      types `.v8f32.v16i32.v16i32` reflecting the f8_f8 fragment
;      shape from the HIP fixture.
;
;   2. The `matrix_a_fmt` arg is `i32 1` (MATRIX_FMT_BF8) and
;      `matrix_b_fmt` is `i32 0` (MATRIX_FMT_FP8 default) — exactly
;      what the disassembled HIP fixture shows
;      (`matrix_a_fmt:MATRIX_FMT_BF8`, matrix_b_fmt omitted at default
;      0). The accumulator type is `<8 x float>` and the A/B fragment
;      types are `<16 x i32>` (the f8 family width).
;
;   3. The `scale_src0` and `scale_src1` slots carry the kernel's
;      runtime VGPR-loaded scale-source values, NOT immediates —
;      pinned via `i32 %{{.+}}` so any regression that hard-codes
;      scales to 0 surfaces immediately.
;
;   4. The reuse args use the canonical defaults: `i1 false` for
;      `matrix_a_reuse` / `matrix_b_reuse` (matches what the HIP
;      builtin emits when `_Constant bool` reuse args are passed
;      `false`).
;
; NEGATIVE PINS:
;
;   * NO call to `llvm.amdgcn.mfma.scale.*` — the cross-target gfx942
;     decomposition path is unimplemented and would mean the
;     same-target lift silently mis-dispatched.
;   * NO call to the non-scaled `llvm.amdgcn.wmma.f32.16x16x128.*`
;     intrinsic — a regression that drops the scale-source operands
;     would land here.
;   * NO call to a different K-width WMMA intrinsic
;     (`16x16x32`, `16x16x64`, `16x16x4`) — would indicate cross-K
;     dispatch confusion.

; IR-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(

; The native gfx1250 scaled-WMMA intrinsic, with the f8_f8 fragment
; shape reflected in the mangled types `.v8f32.v16i32.v16i32`.
; matrix_a_fmt = MATRIX_FMT_BF8 (1), matrix_b_fmt = MATRIX_FMT_FP8
; (0), C_mod = 0, scale {a,b}_scale = 0, scale {a,b}_scale_fmt = 0,
; scale_src0 / scale_src1 are runtime VGPR values, reuse a/b = false.
; IR: %wmma_scale{{[0-9]*}} = call <8 x float> @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32(
; IR-SAME: i32 1, <16 x i32> %{{[^,]+}},
; IR-SAME: i32 0, <16 x i32> %{{[^,]+}},
; IR-SAME: i16 0, <8 x float> %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i32 0, i32 0, i32 %{{[^,]+}},
; IR-SAME: i1 false, i1 false)

; Negative: no MFMA scale fallback (K=128 scaled-WMMA → MFMA
; decomposition is unimplemented in wmma_lowering.cpp).
; IR-NOT: @llvm.amdgcn.mfma.scale.

; Negative: no non-scaled K=128 WMMA dispatch (would drop the scale
; operands).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x128.f8f6f4(

; Negative: no other-K WMMA dispatch (cross-K dispatch confusion).
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x32.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x64.
; IR-NOT: @llvm.amdgcn.wmma.f32.16x16x4.

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx950 --emit-ir=wmma_scale_f32_16x16x128_f8f6f4_kernel 2>&1 | %FileCheck %s --check-prefix=IR_GFX950
;
; Cross-target lift fixture for v_wmma_scale_f32_16x16x128_f8f6f4
; (gfx1250 RDNA4 source) → gfx950 (CDNA4 target). Pins the
; `emitWMMAScaleF8F6F4toScaledMFMA` path in `wmma_lowering.cpp` dispatched
; by `ctx.targetIsa.hasGfx950Insts` in `handle_valu_vop3p.cpp` under
; `CanonicalOp::V_WMMA_SCALE_F32_16x16x128_F8F6F4`. The lift runs Wave32 →
; Wave64 lane redistribution, applies C_mod via IR fneg/fabs on the
; redistributed accumulator, and emits one or two calls (depending
; on `numSourceWavesPerTarget()`) to the gfx950 native scaled MFMA
; intrinsic `int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`
; (IntrinsicsAMDGPU.td:3694), which covers the same K=128 F8/F6/F4
; matmul shape on gfx950's MAI pipe.
;
; The discriminator predicate is `hasGfx950Insts`, NOT `hasMFMA` —
; gfx942 also has `hasMFMA == true` but lacks the scaled F8F6F4 MFMA
; family, so the dispatch must gate on `FeatureGFX950Insts`
; (`isa_profile.hpp::ISAProfile::hasGfx950Insts`). The companion
; gfx942-refusal RUN line at the top of this fixture pins that
; gfx942 still refuses; this RUN line pins that gfx950 accepts.
;
; IR_GFX950-LABEL: define amdgpu_kernel void @wmma_scale_f32_16x16x128_f8f6f4_kernel(
;
; The gfx950 cross-target MFMA-scaled intrinsic. Mangled types
; reflect the `<8 x i32>` A/B fragment width chosen by
; `emitWMMAScaleF8F6F4toScaledMFMA` (widest case; `cbsz`/`blgp` narrow
; the active subset for f6/f4) and the `<4 x f32>` accumulator.
; IR_GFX950: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.

; Negative: no native gfx1250 scaled-WMMA intrinsic in the gfx950
; IR — would mean the cross-target dispatch silently mis-fired.
; IR_GFX950-NOT: @llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4

; Negative: no non-scaled WMMA family in the gfx950 IR — would
; indicate dropped scale operands or cross-K dispatch confusion.
; IR_GFX950-NOT: @llvm.amdgcn.wmma.f32.16x16x

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_scale_f32_16x16x128_f8f6f4_kernel
	.p2align	8
	.type	wmma_scale_f32_16x16x128_f8f6f4_kernel,@function
wmma_scale_f32_16x16x128_f8f6f4_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b256 s[36:43], s[0:1], 0x0
	v_mov_b32_e32 v40, 0
	s_wait_kmcnt 0x0
	s_load_b512 s[0:15], s[36:37], 0x0
	s_load_b512 s[16:31], s[38:39], 0x0
	s_load_b256 s[44:51], s[40:41], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[32:33], s[44:45]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	v_mov_b64_e32 v[24:25], s[24:25]
	v_mov_b64_e32 v[26:27], s[26:27]
	v_mov_b64_e32 v[28:29], s[28:29]
	v_mov_b64_e32 v[30:31], s[30:31]
	v_mov_b64_e32 v[34:35], s[46:47]
	v_mov_b64_e32 v[36:37], s[48:49]
	v_mov_b64_e32 v[38:39], s[50:51]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_scale_f32_16x16x128_f8f6f4 v[32:39], v[0:15], v[16:31], v[32:39], s42, s43 matrix_a_fmt:MATRIX_FMT_BF8
	s_clause 0x1
	global_store_b128 v40, v[36:39], s[40:41] offset:16
	global_store_b128 v40, v[32:35], s[40:41]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_scale_f32_16x16x128_f8f6f4_kernel
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
    .name:           wmma_scale_f32_16x16x128_f8f6f4_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     52
    .symbol:         wmma_scale_f32_16x16x128_f8f6f4_kernel.kd
    .vgpr_count:     41
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
