; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1151 --emit-ir=wmma_f32_16x16x32_f16_kernel 2>/dev/null | %FileCheck %s
;
; NOTE: --isa=gfx1250 is given explicitly. raise_cli otherwise infers
; the SOURCE ISA from the first `gfx<digits>` token in the input path
; (see autoDetectIsa in raise_cli.cpp), and this fixture's name carries
; the TARGET token `gfx1151`, which would mislead that heuristic. The
; real code object is gfx1250 (set by `-mcpu=gfx1250` above).
;
; Lift test for v_wmma_f32_16x16x32_f16 (gfx1250 RDNA4 VOP3P opcode,
; K=32) lowered to gfx1151 (RDNA3.5 / gfx11 family, K=16) via
; emitWMMAtoGFX11WMMA(..., F16).  See:
;   - SemOp::V_WMMA_F32_16x16x32_F16 in transpiler/semop.hpp;
;   - the dispatch in handle-valu-vop3p.cpp guarded by
;     `Ctx.TargetIsa.HasWmma16x16x16F16` (the per-shape capability bit
;     set in isa-profile.cpp for `isGFX11Plus(STI) && isWave32()`);
;   - the lowering body emitWMMAtoGFX11WMMA in wmma-lowering.cpp;
;   - the design rationale in docs/gfx11-wmma-target.design.md.
;
; This is the SAME-WAVE sibling of the gfx942 MFMA lift
; (wmma_f32_16x16x32_bf16.s): gfx1151 is wave32 exactly like gfx1250,
; so there is NO wave-size projection here — the K=32 fragment is
; decomposed and redistributed entirely inside one Wave32.
;
; INVARIANTS PINNED:
;
;   1. The K=32 input decomposes into 2 gfx11 WMMA(K=16) calls with
;      the accumulator chained (klo -> khi). Two calls, not one (a
;      single call would mean a dropped K-tile) and not more (extra
;      calls mean spurious emits). The gfx11 intrinsic is the K=16
;      `llvm.amdgcn.wmma.f32.16x16x16.f16` — NOT the native gfx1250
;      K=32 `llvm.amdgcn.wmma.f32.16x16x32.f16` (target lacks
;      hasTensorOps) and NOT any MFMA (target lacks hasMFMA).
;
;   2. gfx11 RDNA3 A/B fragments are <16 x half> (the lane-half
;      duplicated layout). The lowering bitcasts each redistributed
;      A/B fragment to <16 x half> before the WMMA call.
;
;   3. The accumulator is <8 x float> on both gfx1250 and gfx11, but
;      the IN-REGISTER element->lane mapping differs: gfx1250 C/D is
;      BLOCK (row = 8*(lane/16)+e), gfx11 C/D is INTERLEAVED
;      (row = 2*e + lane/16). The lowering therefore bridges the
;      incoming C accumulator block->interleave BEFORE the WMMA chain
;      and bridges the outgoing D interleave->block AFTER it. Both
;      bridges are ds_bpermute redistributions selected on the
;      lane-half predicate `icmp uge i32 %lane_id, 16`.
;
;   4. Each bridged dword and each WMMA accumulator is wrapped in a
;      whole-wave-mode marker (`llvm.amdgcn.strict.wwm.*`) so the
;      cross-lane redistribute stays correct under a partial-wave
;      (blockDim < 32) launch. Unlike the gfx942 path this needs NO
;      init_whole_wave: there is no wave64 widening, only intra-wave32
;      bpermutes, which strict.wwm alone makes partial-wave-safe.
;
; NEGATIVE PINS:
;
;   * NO native gfx1250 K=32 intrinsic `wmma.f32.16x16x32.f16` — that
;     would mean the lift wrongly believed the target had tensor ops.
;   * NO bf16 intrinsic `wmma.f32.16x16x16.bf16` — would mean the F16
;     SemOp fell through to the BF16 lowering.
;   * NO MFMA of any kind — that is the gfx942 path, not gfx11.
;   * NO init_whole_wave — gfx1151 is wave32, no projection needed.

; CHECK-LABEL: define amdgpu_kernel void @wmma_f32_16x16x32_f16_kernel(

; A/B fragment redistribution: lane-half base addresses for the
; broadcast bpermutes (klo from lanes 0..15, khi from lanes 16..31).
; CHECK: %lane16 = and i32 %lane_id, 15
; CHECK: %klo_addr = shl i32 %lane16, 2
; CHECK: %khi_lane = add i32 %lane16, 16
; CHECK: %khi_addr = shl i32 %khi_lane, 2
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %klo_addr, i32 %{{.*}})
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %khi_addr, i32 %{{.*}})

; C accumulator block->interleave bridge: half-predicate select over
; a bpermute pair, wrapped in whole-wave mode.
; CHECK: %is_high = icmp uge i32 %lane_id, 16
; CHECK: %c_brg = select i1 %is_high, i32 %{{.*}}, i32 %{{.*}}
; CHECK: %c_brg_wwm = call i32 @llvm.amdgcn.strict.wwm.i32(i32 %c_brg)

; The two chained gfx11 K=16 WMMA calls. Both consume <16 x half>
; A/B and an <8 x float> accumulator; the second's accumulator is
; the whole-wave-mode result of the first (K-tile chaining).
; CHECK: %wmma_klo = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half> %{{[^,]+}}, <16 x half> %{{[^,]+}}, <8 x float> %{{[^,]+}})
; CHECK: %wmma_klo_wwm = call <8 x float> @llvm.amdgcn.strict.wwm.v8f32(<8 x float> %wmma_klo)
; CHECK: %wmma_khi = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16(<16 x half> %{{[^,]+}}, <16 x half> %{{[^,]+}}, <8 x float> %wmma_klo_wwm)
; CHECK: %wmma_khi_wwm = call <8 x float> @llvm.amdgcn.strict.wwm.v8f32(<8 x float> %wmma_khi)

; D accumulator interleave->block bridge after the chain (named
; %d_brg*, selected on the recomputed lane-half predicate).
; CHECK: %d_brg{{[0-9]*}} = select i1 %is_high{{[0-9]*}}, i32 %{{.*}}, i32 %{{.*}}

; Negative pins.
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x32.f16
; CHECK-NOT: @llvm.amdgcn.wmma.f32.16x16x16.bf16
; CHECK-NOT: @llvm.amdgcn.mfma
; CHECK-NOT: @llvm.amdgcn.init.whole.wave


	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f32_16x16x32_f16_kernel
	.p2align	8
	.type	wmma_f32_16x16x32_f16_kernel,@function
wmma_f32_16x16x32_f16_kernel:
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
	v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f32_16x16x32_f16_kernel
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
    .name:           wmma_f32_16x16x32_f16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_f32_16x16x32_f16_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
