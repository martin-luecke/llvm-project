; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=v_div_scale_f32_degenerate_xxx_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Regression guard for `v_div_scale_f32`'s degenerate (x, x, x) operand
; shape where src0 == src1 == src2 (all three carry the same register).
; This shape appears in Triton's layer-norm kernel emitted for DiT-XL-2-256
; (triton_red_fused_add_convolution_native_layer_norm_transpose_view_4).
;
; The hardware `v_div_scale_f32 dst, sdst, v_x, v_x, v_x` represents a
; degenerate x/x divide whose ratio is 1.0 for any finite non-zero input.
; Because numer == denom the scale-flag convention is irrelevant for the
; downstream div_fixup result, so the decoder treats the shape as
; scale-numerator (flag = true) to keep the LLVM intrinsic call well-formed.
;
; Prior to the fix the three-arm matcher's `else` arm fired for
; Src0EqSrc1 && Src0EqSrc2, emitting an "operand triple does not match a
; known divide-scaling shape" failure and refusing the entire kernel.
;
; This fixture pins two stable anchors:
;
;   (a) the degenerate scale call is lifted as scale-NUMERATOR:
;       `@llvm.amdgcn.div.scale.f32(float %N, float %N, i1 true)` --
;       the first and second arguments are the same bitcast (same VGPR
;       occupies both numer and denom slots), and the flag is `i1 true`.
;
;   (b) a negative assertion: the pre-fix error string must NOT appear in
;       the raised IR -- confirming the kernel was accepted, not refused.

; CHECK-LABEL: define amdgpu_kernel void @v_div_scale_f32_degenerate_xxx_kernel(

; (a) The degenerate (x, x, x) scale call must be lifted as scale-numerator.
;     Both the numer and denom arguments alias the same VGPR value, and the
;     i1 flag must be `true` (scale-numerator convention).
; CHECK: call { float, i1 } @llvm.amdgcn.div.scale.f32(float %{{[^,]+}}, float %{{[^,]+}}, i1 true)

; (b) Negative: the raiser must not emit the "does not match a known
;     divide-scaling shape" refusal error as IR text.
; CHECK-NOT: operand triple does not match

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_div_scale_f32_degenerate_xxx_kernel
	.p2align	8
	.type	v_div_scale_f32_degenerate_xxx_kernel,@function
v_div_scale_f32_degenerate_xxx_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	s_wait_kmcnt 0x0
	flat_load_b32 v1, v0, s[6:7] scope:SCOPE_SYS
	s_wait_loadcnt_dscnt 0x0
	; Degenerate x/x shape: src0 == src1 == src2 (all carry the same VGPR).
	; This is the operand triple that DiT-XL-2-256 emits and that triggered
	; the "does not match a known divide-scaling shape" failure before the fix.
	v_div_scale_f32 v2, vcc_lo, v1, v1, v1
	; Minimal Newton-iteration chain so the kernel is structurally complete.
	v_rcp_f32_e32 v3, v2
	v_nop
	v_fma_f32 v4, -v2, v3, 1.0
	v_fmac_f32_e32 v3, v4, v3
	v_mul_f32_e32 v5, v2, v3
	v_fma_f32 v6, -v2, v5, v2
	v_fmac_f32_e32 v5, v6, v3
	v_fma_f32 v2, -v2, v5, v2
	v_div_fmas_f32 v2, v2, v3, v5
	v_div_fixup_f32 v1, v2, v1, v1
	global_store_b32 v0, v1, s[4:5]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_div_scale_f32_degenerate_xxx_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 8
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           v_div_scale_f32_degenerate_xxx_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         v_div_scale_f32_degenerate_xxx_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
