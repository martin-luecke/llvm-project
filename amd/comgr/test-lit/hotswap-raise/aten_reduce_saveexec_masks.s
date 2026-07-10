; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=aten_reduce_saveexec_masks_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; ATen reductions use scalar SAVEEXEC around ordinary VCC/V_CMP bounds masks
; and EXEC restore fragments near mbcnt-driven ds_bpermute reductions.  These
; masks are safe when the source reaches SAVEEXEC as an EXEC-width wave-mask
; fact; this fixture pins the three local shapes seen in the Qwen reduce window.

; CHECK-LABEL: define amdgpu_kernel void @aten_reduce_saveexec_masks_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	aten_reduce_saveexec_masks_kernel
	.p2align	8
	.type	aten_reduce_saveexec_masks_kernel,@function
aten_reduce_saveexec_masks_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0

; Direct VCC source.
; CHECK: %vcmp = icmp ugt i32 32, %{{.*}}
	v_cmp_gt_u32_e32 vcc_lo, 32, v0
; CHECK: %vcc_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %vcmp)
; CHECK: %new_exec = and i64 %saved_exec, %vcc_ballot
	s_and_saveexec_b32 s2, vcc_lo
	v_mov_b32_e32 v2, 1
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_mov_b32 exec_lo, s2

; SOP2 VCC-and-SGPR mask algebra source.
	v_cmp_gt_u32_e32 vcc_lo, 16, v0
	v_cmp_gt_u32_e64 s2, 48, v0
; CHECK: %wave_mask_and = and i1
; CHECK: %wave_mask_and_scc_ballot = call i64 @llvm.amdgcn.ballot.i64(i1 %wave_mask_and)
	s_and_b32 s3, vcc_lo, s2
; CHECK: %exec_width_sgpr_shadow_sel{{[0-9]*}} = select i1 true, i64 %wm_shadow_exec{{[0-9]*}}, i64 %wn_src_to_exec_mask{{[0-9]*}}
; CHECK: %new_exec{{[0-9]*}} = and i64 %{{.*}}, %exec_width_sgpr_shadow_sel{{[0-9]*}}
	s_and_saveexec_b32 s4, s3
	v_mov_b32_e32 v2, 2
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_mov_b32 exec_lo, s4

; EXEC save/xor restore fragment source.
	s_mov_b32 s2, exec_lo
	v_cmpx_ne_u32_e64 v0, 0
; CHECK: %wave_mask_xor = xor i1
	s_xor_b32 s3, exec_lo, s2
; CHECK: %exec_width_sgpr_shadow_sel{{[0-9]*}} = select i1 true, i64 %wm_shadow_exec{{[0-9]*}}, i64 %wn_src_to_exec_mask{{[0-9]*}}
; CHECK: %new_exec{{[0-9]*}} = and i64 %exec_width_sgpr_shadow_sel{{[0-9]*}},
	s_and_not1_saveexec_b32 s4, s3
	v_mov_b32_e32 v2, 3
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_or_b32 exec_lo, exec_lo, s4

	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel aten_reduce_saveexec_masks_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 5
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           aten_reduce_saveexec_masks_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     5
    .symbol:         aten_reduce_saveexec_masks_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
