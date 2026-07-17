; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ds_swizzle_b32_phantom_modrep_neutralize_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Confirm this kernel takes the phantom-lane MODREP fallback (not WaveNative):
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ds_swizzle_b32_phantom_modrep_neutralize_kernel 2>&1 >/dev/null \
; RUN:   | %FileCheck %s --check-prefix=PROJ
;
; PROJ: phantom-lane regime
; PROJ-SAME: falling back to ModuloReplicationProjection
;
; rocm-systems#156: a gfx1250 ds_swizzle_b32 BITMASK_PERM butterfly reduction
; (imm 0x041f == SWAP-1, bit 15 = 0) whose result feeds a data-dependent address.
; ds_swizzle is convergent and runs on all 64 physical lanes; under the phantom-
; lane MODREP fallback (max_flat_workgroup_size 32 < wave64) the upper target
; lanes are hardware-inactive and hold undef VGPRs. Without neutralisation their
; undef flows through the butterfly into an active lane's result, and here that
; result is folded into a global load/store address -- a wild pointer -> GPU
; memory fault (the pinned #120 Wan layer-norm reduction).
;
; The fix forces the swizzle INPUT to 0 on the undispatched phantom lanes
; (flat lane id >= max_flat_workgroup_size) via a plain per-lane select before
; the convergent swizzle, so a phantom lane can only ever contribute a benign 0
; (not undef) through the reduction. A plain select (not set.inactive/WWM) keeps
; codegen light on large reduction kernels. Sibling c2_ds_swizzle.s pins the
; plain (non-phantom, WaveNative-eligible) lift.

; CHECK-LABEL: define amdgpu_kernel void @ds_swizzle_b32_phantom_modrep_neutralize_kernel(
; A lane is real iff its flat lane id is below max_flat_workgroup_size (32 here);
; the undispatched upper lanes are forced to 0 before the convergent swizzle, and
; the neutralised value (not the raw input) is what the swizzle reads. Match the
; swizzle-neutralisation names specifically (the emitWorkitemIdX phantom clamp
; uses the same icmp/select shape earlier in the kernel).
; CHECK: [[REAL:%phantom_neut_is_real_lane[0-9]*]] = icmp ult i32 %{{[^,]+}}, 32
; CHECK-NEXT: [[NEUT:%ds_swiz_neut[0-9]*]] = select i1 [[REAL]], i32 %{{[^,]+}}, i32 0
; CHECK-NEXT: call i32 @llvm.amdgcn.ds.swizzle(i32 [[NEUT]], i32 1055)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_swizzle_b32_phantom_modrep_neutralize_kernel
	.p2align	8
	.type	ds_swizzle_b32_phantom_modrep_neutralize_kernel,@function
ds_swizzle_b32_phantom_modrep_neutralize_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	global_load_b32 v0, v0, s[2:3]
	s_wait_loadcnt 0x0
	ds_swizzle_b32 v0, v0 offset:0x041f
	s_wait_dscnt 0x0
	v_mad_u32 v1, v0, 4, v0
	global_load_b32 v2, v1, s[2:3]
	s_wait_loadcnt 0x0
	global_store_b32 v1, v2, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_swizzle_b32_phantom_modrep_neutralize_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           ds_swizzle_b32_phantom_modrep_neutralize_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_swizzle_b32_phantom_modrep_neutralize_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
