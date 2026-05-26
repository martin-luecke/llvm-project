; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not %raise_cli %t.hsaco --emit-ir 2>&1 | %FileCheck %s --check-prefix=STDERR
;
; Loud-refusal test for buffer_atomic_max_f64. See the sibling
; buffer_atomic_min_f64.s test for the full rationale; mirror-image
; situation (raw `src > tmp ? src : tmp` on gfx942 vs IEEE 754-2019
; maximumNumber on gfx1250, no available LLVM IR shape bit-exact
; for both, refusing rather than emitting approximate IR).

; STDERR: transpiler: Unsupported buffer atomic: buffer_atomic_max_f64
; STDERR: raise_cli: kernel 'buffer_atomic_max_f64_kernel' failed to raise:
; STDERR-SAME: buffer_atomic_max_f64
; STDERR-SAME: [MUBUF]

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_max_f64_kernel
	.p2align	8
	.type	buffer_atomic_max_f64_kernel,@function
buffer_atomic_max_f64_kernel:
	s_load_dwordx4 s[0:3], s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, s3
	;;#ASMSTART
	buffer_atomic_max_f64 v[2:3], v0, s[0:3], 0 offen
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_max_f64_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 4
		.amdhsa_reserve_vcc 1
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           8, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_max_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_atomic_max_f64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
