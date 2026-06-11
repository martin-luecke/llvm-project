; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=andn2_saveexec_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression test for S_ANDN2_SAVEEXEC_B32 / S_ORN2_SAVEEXEC_B32 operand
; order under cross-widening (gfx1250 wave32 -> gfx942 wave64).
;
; Hardware semantics:
;   S_ANDN2_SAVEEXEC: dst = EXEC; EXEC = SRC & ~EXEC
;   S_ORN2_SAVEEXEC:  dst = EXEC; EXEC = SRC | ~EXEC
;
; A prior bug computed EXEC & ~SRC (NOT applied to SRC instead of EXEC),
; which caused the "else" branch of OCML two-path approximations to
; execute with the "if" lane mask. Every kernel using __ocml_tanh_f32,
; __ocml_erf_f32, __ocml_sin_f32, or __ocml_cos_f32 was affected.
;
; This test assembles a minimal if/else kernel that uses the OCML-style
; pattern: v_cmpx narrows exec to "if" lanes, s_xor computes the "else"
; mask, and s_and_not1_saveexec enters the else branch. The CHECK lines
; verify that the raised IR computes SRC & ~EXEC (NOT on the old exec
; value, not on the source operand).

; CHECK-LABEL: define amdgpu_kernel void @andn2_saveexec_kernel(

; s_and_not1_saveexec: hardware semantics are EXEC = SRC & ~EXEC.
; The NOT must be applied to the CURRENT EXEC (cmpx_exec — the "if"
; lanes), then ANDed with SRC (the else-mask from the shadow). A prior
; bug had EXEC & ~SRC which re-ran the "if" lanes instead of "else".
;
; After mem2reg, the else-mask shadow flows through a select:
;   %shadow = select i1 true, i64 %ballot_xor, i64 %widened_sgpr
; Then the NOT+AND for s_and_not1_saveexec:
;   %N = xor i64 %cmpx_exec, -1           ; NOT(current_exec)
;   %new_exec = and i64 %shadow, %N        ; src & ~exec
;
; The critical invariant: NOT is applied to %cmpx_exec (the current
; exec after v_cmpx narrowing), and the AND pairs it with the source
; (the shadow select), producing the "else" lanes. The buggy version
; had AND(%cmpx_exec, NOT(%shadow)) which gave back the "if" lanes.
; CHECK:       %exec_width_sgpr_shadow_sel = select i1 true, i64
; CHECK:       %[[NOT:[0-9]+]] = xor i64 %cmpx_exec, -1
; CHECK-NEXT:  %new_exec{{[0-9]*}} = and i64 %exec_width_sgpr_shadow_sel, %[[NOT]]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	andn2_saveexec_kernel
	.p2align	8
	.type	andn2_saveexec_kernel,@function
andn2_saveexec_kernel:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v2, 0
	s_wait_kmcnt 0x0
	; Guard: only threads with tid < 256 participate
	v_cmp_gt_i32_e32 vcc_lo, 0x100, v0
	s_and_saveexec_b32 s4, vcc_lo
	s_cbranch_execz .Lend
	; Load input
	global_load_b32 v2, v1, s[0:1] scale_offset
	s_wait_loadcnt 0x0

	; if/else pattern (mirrors OCML tanh structure):
	;   s_mov_b32 s5, exec_lo           ; save full exec
	;   v_cmpx_gt_f32 ...               ; narrow to "if" lanes
	;   s_xor_b32 s5, exec_lo, s5       ; compute "else" lanes
	;   <if body: multiply by 2>
	;   s_and_not1_saveexec_b32 s4, s5   ; enter else branch
	;   <else body: multiply by 3>
	;   s_or_b32 exec_lo, exec_lo, s4   ; restore exec
	s_mov_b32 s5, exec_lo
	v_cmpx_gt_f32_e32 0x3f000000, v2
	s_xor_b32 s5, exec_lo, s5
	s_cbranch_execz .Lelse
	; "if" body: v2 = v2 * 2.0 (for lanes where v2 > 0.5)
	v_mul_f32_e32 v2, 2.0, v2
.Lelse:
	s_and_not1_saveexec_b32 s4, s5
	s_cbranch_execz .Ljoin
	; "else" body: v2 = v2 * 3.0 (for lanes where v2 <= 0.5)
	v_mul_f32_e32 v2, 0x40400000, v2
.Ljoin:
	s_or_b32 exec_lo, exec_lo, s4
	; Store result
	global_store_b32 v1, v2, s[2:3] scale_offset
.Lend:
	s_or_b32 exec_lo, exec_lo, s4
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel andn2_saveexec_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .address_space: global, .offset: 8, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 256
    .name: andn2_saveexec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 8
    .symbol: andn2_saveexec_kernel.kd
    .vgpr_count: 3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
