; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:      --enable-lds-redirect --force-lds-redirect \
; RUN:      --emit-ir=lds_redirect_ceildiv_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Pin that emitHiddenBlockCount uses ceiling division (G + W - 1) / W, not
; floor (G / W).  For non-aligned dispatches (grid_size % wg_size != 0),
; floor underestimates the workgroup count by 1, causing the last WG in each
; dimension to alias another WG's scratch region.
;
; The fixture is the simplest LDS-using kernel (4 KiB LDS, 1-arg, 32 threads).
; The IR-level check pins the intermediate value names emitted by the ceildiv
; expansion.  A plain UDiv ("floor") raiser would emit none of them.
;
; Invariants:
;   1. block_count_wm1_{dim} = wg_size[dim] - 1      (the "+W-1" in ceil)
;   2. block_count_num_{dim} = grid_size[dim] + wm1   (numerator before divide)
;   3. source_hidden_block_count_{dim} = num / wg_size  (the ceildiv result)
; Names are dim-qualified so multiple calls to emitHiddenBlockCount produce
; unique IR value names without LLVM's dedup suffixes.

; CHECK-LABEL: define amdgpu_kernel void @lds_redirect_ceildiv_kernel(

; Ceildiv for dim 0 (X): wm1 = wg_size_x - 1, num = grid_x + wm1.
; CHECK: %block_count_wm1_0 =
; CHECK: %block_count_num_0 =
; CHECK: %source_hidden_block_count_0 = udiv

; Ceildiv for dim 1 (Y).
; CHECK: %block_count_wm1_1 =
; CHECK: %block_count_num_1 =
; CHECK: %source_hidden_block_count_1 = udiv

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	lds_redirect_ceildiv_kernel
	.p2align	8
	.type	lds_redirect_ceildiv_kernel,@function
lds_redirect_ceildiv_kernel:
	v_lshlrev_b32_e32 v1, 2, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	ds_store_b32 v1, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	ds_load_b32 v1, v1
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_redirect_ceildiv_kernel
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
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
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           lds_redirect_ceildiv_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         lds_redirect_ceildiv_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
