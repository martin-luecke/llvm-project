; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:   --enable-high-precision-mfma \
; RUN:   --emit-ir=wmma_f32_16x16x32_bf16_high_precision_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for v_wmma_f32_16x16x32_bf16 (gfx1250 RDNA4 VOP3P opcode
; 0x062) lowered to gfx942 (CDNA3) under the OPT-IN high-precision
; MFMA flag (--enable-high-precision-mfma). The default-off path is
; pinned by `wmma_f32_16x16x32_bf16.s`; this fixture pins the IR
; shape that the flag produces so a regression that silently re-routes
; the bf16 path through the default chained-bf16-K=16 MFMA fails this
; test instead of silently producing different numerics.
;
; See `RaiseContext::enableHighPrecisionMfma` and the block comment
; on `wmma_lowering.cpp::emitChainedF32MfmaBF16Upcast` for the
; lowering body and the bit-exactness contract — short version:
; this path is NOT bit-exact equivalent to either the source WMMA
; or to the default chained-bf16 path; it is a more numerically-
; faithful reference for drift triage.
;
; INVARIANTS PINNED (load-bearing semantics only — the fixture
; deliberately does NOT pin specific SSA names or the count of
; MFMA calls beyond what's necessary to catch a regression that
; reverts the high-precision path back to the default chained-bf16
; path):
;
;   1. The dispatched MFMA intrinsic is `mfma.f32.16x16x4f32`. The
;      default-off path emits `mfma.f32.16x16x16bf16.1k`, so seeing
;      a `4f32` call AND not seeing a `16bf16.1k` call together
;      proves the high-precision path was taken.
;
;   2. Software bf16 → fp32 upcast happens before the MFMA. The cast
;      chain is `bitcast i32 → <2 x bfloat>` followed by `fpext
;      <2 x bfloat> → <2 x float>`. Both pieces must appear in the
;      lifted IR; either one alone would be inconclusive.
;
;   3. The MFMA accumulator is chained: at least one MFMA call
;      consumes the result of an earlier MFMA call as its `acc`
;      operand. This pins the K-decomposition shape (chained, not
;      reduced via fadd) so a regression that emitted parallel
;      MFMAs and folded with `fadd` would fail here.
;
;   4. The accumulator type is `<4 x float>`, identical to the
;      default-off bf16 path's MFMA accumulator.
;
;   5. Single `@llvm.amdgcn.init.whole.wave` at function entry
;      (WaveNativeProjection's kernel-wide HW EXEC=-1 invariant).

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x32_bf16_high_precision_kernel(

; Kernel-entry EXEC virtualisation: one init_whole_wave call.
; CHECK: call i1 @llvm.amdgcn.init.whole.wave()

; Software bf16 → fp32 upcast (invariant #2). Order matters: the
; bitcast must precede the fpext; CHECK lines run in source order
; so this also pins the operation order.
; CHECK: bitcast i32 %{{.*}} to <2 x bfloat>
; CHECK: fpext <2 x bfloat> %{{.*}} to <2 x float>

; The K=4 fp32 MFMA (invariant #1, accumulator type pinned by
; invariant #4). Capture the result SSA name so the chained-call
; pattern below can reference it; the name itself is incidental,
; but FileCheck's variable-binding requires capturing it.
; CHECK: %[[FIRST:[a-zA-Z0-9_.]+]] = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{.*}}, float %{{.*}}, <4 x float> %{{.*}}, i32 0, i32 0, i32 0)

; Chained accumulator (invariant #3): a subsequent MFMA call must
; take the previous MFMA's result as its `acc` operand.
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %{{.*}}, float %{{.*}}, <4 x float> %[[FIRST]]{{.*}}, i32 0, i32 0, i32 0)

; Negative pin: the default chained-bf16 MFMA path must NOT appear
; under the flag — that would mean the flag silently fell through
; to the default path (which would emit `mfma.f32.16x16x16bf16.1k`).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16bf16.1k

; Negative pin: the F16 sibling intrinsic must NOT appear (would
; mean BF16 dispatch fell through to F16).
; CHECK-NOT: @llvm.amdgcn.mfma.f32.16x16x16f16

; Single `init.whole.wave` call per kernel.
; CHECK-NOT: call {{.*}} @llvm.amdgcn.init.whole.wave


	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f32_16x16x32_bf16_high_precision_kernel
	.p2align	8
	.type	wmma_f32_16x16x32_bf16_high_precision_kernel,@function
wmma_f32_16x16x32_bf16_high_precision_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[24:27], s[0:1], 0x0
	s_load_b64 s[28:29], s[0:1], 0x10
	v_mov_b32_e32 v24, 0
	s_wait_kmcnt 0x0
	s_load_b256 s[0:7], s[24:25], 0x0
	s_load_b256 s[8:15], s[26:27], 0x0
	s_load_b256 s[16:23], s[28:29], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f32_16x16x32_bf16_high_precision_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 30
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           wmma_f32_16x16x32_bf16_high_precision_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_f32_16x16x32_bf16_high_precision_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
