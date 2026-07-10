; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=mbcnt_saveexec_mask_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:     --emit-ir=mbcnt_saveexec_mask_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MODREP
;
; Wave32 scalar SAVEEXEC masks derived from source-wave-local mbcnt are safe
; under WaveNative only when the mask reaches the SAVEEXEC handler at EXEC
; width.  The V_CMP -> SGPR path records a full target-width shadow, and
; `readOpExecWidth` must select that shadow instead of the lossy source-width
; fallback before `storeExec` commits the new EXEC value.

; WN-LABEL: define amdgpu_kernel void @mbcnt_saveexec_mask_kernel(
; MODREP: cross-wave-lane-predicated-exec
; MODREP: SaveExecFromLaneId
; MODREP: outcome: (c) refuse

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	mbcnt_saveexec_mask_kernel
	.p2align	8
	.type	mbcnt_saveexec_mask_kernel,@function
mbcnt_saveexec_mask_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
; WN: %exec_srcwave_mask_base = and i32 %{{[^,]+}}, -32
; WN: %mbcnt_masked{{[0-9]*}} = and i32 %exec_srcwave_mask, %mbcnt_below_mask{{[0-9]*}}
; WN: %[[MBCNT:.*]] = phi i32 [ %mbcnt_lo_srcwave{{[0-9]*}},
	v_mbcnt_lo_u32_b32 v1, exec_lo, 0
	v_mbcnt_hi_u32_b32 v1, exec_hi, v1

; WN: %[[CMP:.*]] = icmp ult i32 %[[MBCNT]], 16
	v_cmp_lt_u32_e64 s2, v1, 16
; WN: %wm_shadow_exec{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMP]])
; WN: %exec_width_sgpr_shadow_sel{{[0-9]*}} = select i1 true, i64 %wm_shadow_exec{{[0-9]*}}, i64 %wn_src_to_exec_mask{{[0-9]*}}
; WN: %new_exec = and i64 %saved_exec, %exec_width_sgpr_shadow_sel{{[0-9]*}}
	s_and_saveexec_b32 s3, s2
	v_mov_b32_e32 v2, 1
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_mov_b32 exec_lo, s3

; WN: %[[CMP2:.*]] = icmp ugt i32 %{{.*}}, 7
	v_cmp_gt_u32_e64 s2, v1, 7
; WN: %wm_shadow_exec{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMP2]])
; WN: %new_exec{{[0-9]*}} = and i64 %exec_width_sgpr_shadow_sel{{[0-9]*}},
	s_and_not1_saveexec_b32 s3, s2
	v_mov_b32_e32 v2, 2
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_or_b32 exec_lo, exec_lo, s3

; WN: %[[CMP3:.*]] = icmp eq i32 %{{.*}}, 0
	v_cmp_eq_u32_e64 s2, v1, 0
; WN: %wm_shadow_exec{{[0-9]*}} = call i64 @llvm.amdgcn.ballot.i64(i1 %[[CMP3]])
; WN: %new_exec{{[0-9]*}} = or i64 %{{.*}}, %exec_width_sgpr_shadow_sel{{[0-9]*}}
	s_or_saveexec_b32 s3, s2
	v_mov_b32_e32 v2, 3
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_mov_b32 exec_lo, s3

	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mbcnt_saveexec_mask_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           mbcnt_saveexec_mask_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         mbcnt_saveexec_mask_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
