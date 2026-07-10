; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=mbcnt_saveexec_mask_shift_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SHIFT
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=mbcnt_saveexec_mask_one_sided_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=ONESIDED
;
; Lane-position-sensitive masks must not fall back to source-width widening at
; SAVEEXEC.  If scalar algebra destroys the EXEC-width shadow, WaveNative refuses
; at the SAVEEXEC site rather than replicating low32 into both packed waves.

; SHIFT: cross-wave-lane-predicated-exec
; SHIFT: s_and_saveexec_b32
; SHIFT: SAVEEXEC source mask lacks an EXEC-width wave-mask proof
; ONESIDED: cross-wave-lane-predicated-exec
; ONESIDED: s_and_saveexec_b32
; ONESIDED: SAVEEXEC source mask lacks an EXEC-width wave-mask proof

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	mbcnt_saveexec_mask_shift_refuse_kernel
	.p2align	8
	.type	mbcnt_saveexec_mask_shift_refuse_kernel,@function
mbcnt_saveexec_mask_shift_refuse_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	v_mbcnt_hi_u32_b32 v1, exec_hi, v1
	v_cmp_lt_u32_e64 s2, v1, 16
	s_lshr_b32 s2, s2, 1
	s_and_saveexec_b32 s3, s2
	v_mov_b32_e32 v2, 1
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm

	.globl	mbcnt_saveexec_mask_one_sided_refuse_kernel
	.p2align	8
	.type	mbcnt_saveexec_mask_one_sided_refuse_kernel,@function
mbcnt_saveexec_mask_one_sided_refuse_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	v_mbcnt_hi_u32_b32 v1, exec_hi, v1
	v_cmp_lt_u32_e64 s2, v1, 16
	s_lshr_b32 s4, s2, 1
	s_and_b32 s3, s2, s4
	s_and_saveexec_b32 s5, s3
	v_mov_b32_e32 v2, 1
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_saveexec_mask_shift_refuse_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_saveexec_mask_one_sided_refuse_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
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
    .name:           mbcnt_saveexec_mask_shift_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         mbcnt_saveexec_mask_shift_refuse_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mbcnt_saveexec_mask_one_sided_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         mbcnt_saveexec_mask_one_sided_refuse_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
