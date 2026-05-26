; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=add_pc_i64_kernel 2>/dev/null | %FileCheck %s
;
; Lift test for s_add_pc_i64 (gfx1250 SOP1 PC-relative direct branch).
; The handler in handle-sop1.cpp lowers the immediate-literal form to
;   br label %bb_<target>
; where target = site_offset + site_size + signed_i64_imm.
;
; For this fixture:
;   site_offset = 0x08 (after the 8-byte s_load_b64)
;   site_size   = 0x04 (BE804B88 -- inline-constant 8 fits in the SOP1
;                       4-byte encoding)
;   imm         = 8    (skip the 8-byte v_mov_b32 with 32-bit literal)
;   target      = 0x14 (s_wait_kmcnt 0x0)
;
; Block-leader discovery is special-cased in decode.cpp's
; collectBranchTargets (the legacy 16-bit short-branch decode would
; mis-truncate the i64 operand). A regression that loses the special
; case would either insert no leader (handler bails to a fallback BB)
; or insert a garbage one.

; CHECK-LABEL: define amdgpu_kernel void @add_pc_i64_kernel(
; CHECK: br label %bb_0x14
; CHECK-NOT: indirectbr
; CHECK-NOT: unreachable
; CHECK: bb_0x14:
; The dead v_mov payload at 0x0C-0x14 must not surface as an IR
; constant -- a regression that fell through into it would emit the
; literal 0xDEAD0001 (3735618049) into the lifted IR.
; CHECK-NOT: 3735618049
; CHECK-NOT: 0xDEAD0001

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	add_pc_i64_kernel
	.p2align	8
	.type	add_pc_i64_kernel,@function
add_pc_i64_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_add_pc_i64 8
	v_mov_b32 v1, 0xDEAD0001
	;;#ASMEND
	s_wait_kmcnt 0x0
	v_mov_b32 v1, 0xCAFE0002
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel add_pc_i64_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           add_pc_i64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         add_pc_i64_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
