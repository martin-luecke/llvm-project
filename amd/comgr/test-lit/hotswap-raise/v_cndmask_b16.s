; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_cndmask_b16_kernel | %FileCheck %s
;
; Lift test for gfx1250 true16 v_cndmask_b16 (VOP3-only). Semantically the
; 16-bit sibling of v_cndmask_b32: pick src1 or src0 per the per-lane wave-mask
; condition (src2), but on a 16-bit half of the dst, preserving the other half
; (RDNA3+ true16). op_sel routes the src0/src1 halves (lo/lo here). The per-lane
; condition read is shared with the b32 lift (raiseCndmaskWaveCondition), so
; this fixture just pins the 16-bit select + half-merge shape.

; CHECK-LABEL: define amdgpu_kernel void @v_cndmask_b16_kernel(
; The per-lane compare feeds the select condition.
; CHECK: [[CMP:%.+]] = icmp ult i32 0,
; Both 16-bit source halves are truncated out of their containing dword.
; CHECK: trunc i32 {{.+}} to i16
; CHECK: trunc i32 {{.+}} to i16
; The select is on i16 operands, gated by the compare.
; CHECK: [[SEL:%.+]] = select i1 [[CMP]], i16 {{.+}}, i16
; Merge back into the low half of the dst (upper half preserved).
; CHECK: zext i16 [[SEL]] to i32
; CHECK: and i32 {{.+}}, -65536
; CHECK: or i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_cndmask_b16_kernel
	.p2align	8
	.type	v_cndmask_b16_kernel,@function
v_cndmask_b16_kernel:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v1, s0
	v_mov_b32_e32 v2, s1
	v_cmp_lt_u32_e64 s0, 0, v0
	v_cndmask_b16 v1.l, v1.l, v2.l, s0
	global_store_b32 v0, v1, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_cndmask_b16_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
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
    .name:           v_cndmask_b16_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         v_cndmask_b16_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
