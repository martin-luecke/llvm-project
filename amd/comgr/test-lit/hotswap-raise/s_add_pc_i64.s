; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=s_add_pc_i64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Lift test for `s_add_pc_i64 imm64` -- gfx1250/gfx13 PC-relative
; unconditional long-branch trampoline.
; PC_next = PC_after_inst + sign_extend(imm64). Emitted as a trampoline
; immediately after a conditional skip to extend `s_cbranch_<cond>` past
; the +/-128KiB simm16 window:
;
;     s_cbranch_<cond> 1
;     s_add_pc_i64 imm64
;
; The lift must emit an unconditional `br` to the resolved target BB,
; not `unreachable` (handler refused) or `indirectbr` (mis-classified
; as register-indirect).
;
; CHECK-LABEL: define amdgpu_kernel void @s_add_pc_i64_kernel(
; CHECK: br label %bb_0x{{[0-9a-f]+}}
; CHECK-NOT: unreachable
; CHECK-NOT: indirectbr ptr

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_add_pc_i64_kernel
	.p2align	8
	.type	s_add_pc_i64_kernel,@function
s_add_pc_i64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	;;#ASMSTART
	s_cbranch_scc1 1
	s_add_pc_i64 0
	v_mov_b32 v1, 0xDEAD0001

	;;#ASMEND
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_add_pc_i64_kernel
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
    .name:           s_add_pc_i64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         s_add_pc_i64_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
