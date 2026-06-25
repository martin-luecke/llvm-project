; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=divscale_flag_vcc_hi_chain_kernel 2>/dev/null | %FileCheck %s
;
; Wave32 source raised to a wave64 target.  Exercises the wave32 fdiv
; expansion that parks a v_div_scale_f32 flag (a genuine per-lane mask) in the
; vcc_hi scratch scalar and then restores it to VCC via `s_mov_b32 vcc_lo,
; vcc_hi` immediately before the consuming v_div_fmas -- the shape SD3.5's
; layer-norm kernel emits.  writeCarryOutI1 records a same-BB per-lane i1
; shadow keyed on the vcc_hi scratch slot (handle-valu.cpp), and handleSOP1's
; `s_mov_b32 vcc_lo, vcc_hi` reader (handle-sop1.cpp) commits that exact i1
; straight into VCC instead of re-widening the truncated 32-bit data slot
; (which would drop lanes 32..63 on wave64).  The SGPR-slot variant of the
; same idiom is covered by v_div_scale_carry_sgpr_chain_wave32.s.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	divscale_flag_vcc_hi_chain_kernel
	.p2align	8
	.type	divscale_flag_vcc_hi_chain_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @divscale_flag_vcc_hi_chain_kernel(
divscale_flag_vcc_hi_chain_kernel:
	v_div_scale_f32 v3, vcc_hi, v1, v1, v0
	; Producer's per-lane flag i1, captured directly from div_scale:
	; CHECK: %divscale = call { float, i1 } @llvm.amdgcn.div.scale.f32(
	; CHECK: [[FLAG:%[0-9]+]] = extractvalue { float, i1 } %divscale, 1
	s_mov_b32 vcc_lo, vcc_hi
	v_div_fmas_f32 v3, v3, v1, v0
	; The vcc_hi scratch shadow preserves the full-width flag, so the fmas
	; consumes the producer's per-lane i1 directly -- no lossy re-widening:
	; CHECK-NOT: mask_widen
	; CHECK: call float @llvm.amdgcn.div.fmas.f32(float %{{.+}}, float %{{.+}}, float %{{.+}}, i1 [[FLAG]])
	ds_store_b32 v4, v3
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel divscale_flag_vcc_hi_chain_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           divscale_flag_vcc_hi_chain_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         divscale_flag_vcc_hi_chain_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
