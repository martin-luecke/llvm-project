; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1151 \
; RUN:       --emit-ir=wmma_f32_16x16x64_fp8_fp8_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
;
; NOTE: --isa=gfx1250 is given explicitly so raise_cli does not infer
; the SOURCE ISA from the `gfx1151` (target) token in this fixture's
; filename. The real code object is gfx1250.
;
; Negative companion to wmma_f32_16x16x32_f16_gfx1151.s. The gfx11
; WMMA lowering (emitWMMAtoGFX11WMMA) is deliberately SCOPED to the
; 16-bit float shapes only — F16 and BF16 — because gfx11 hardware
; provides no fp8/bf8 WMMA. The capability bit HasWmma16x16x16F16 is
; therefore set for those two SemOps only, and the FP8 SemOp must
; fall through to the refusal arm of the dispatch in
; handle-valu-vop3p.cpp rather than emit IR that cannot be lowered.
;
; This pins the SCOPE boundary: a regression that widened the gfx11
; path to fp8 (e.g. by gating on a target-family check instead of the
; per-shape capability bit, contrary to matrix-translation.md §5.0.1
; "branch on a flag, not a target triple") would make this kernel
; raise successfully and this test would fail.

; REFUSE: failed to raise
; REFUSE-SAME: v_wmma_f32_16x16x64_fp8_fp8
; REFUSE: has no available lowering on the target ISA
; REFUSE: refusing rather than emitting IR that cannot be lowered


	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_f32_16x16x64_fp8_fp8_kernel
	.p2align	8
	.type	wmma_f32_16x16x64_fp8_fp8_kernel,@function
wmma_f32_16x16x64_fp8_fp8_kernel:
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
	v_wmma_f32_16x16x64_fp8_fp8 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_f32_16x16x64_fp8_fp8_kernel
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
    .name:           wmma_f32_16x16x64_fp8_fp8_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_f32_16x16x64_fp8_fp8_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
