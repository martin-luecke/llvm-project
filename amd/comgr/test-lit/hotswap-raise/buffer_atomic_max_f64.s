; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>/dev/null | %FileCheck %s
;
; CAS-loop lift test for buffer_atomic_max_f64 from a gfx942 source
; binary. Mirror of buffer_atomic_min_f64.s; see that fixture for
; the full rationale on why a CAS loop is the only accurate lift.
; gfx942 (ISA manual 12.15.3 op 81) is raw `src > tmp ? src : tmp`,
; lowered here via `fcmp ogt` + `select` inside a
; raw_buffer_atomic_cmpswap.i64 loop.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_max_f64_kernel(
; CHECK: %fp64_minmax_init = call i64 @llvm.amdgcn.raw.buffer.load.i64
; CHECK: fp64_minmax_loop:
; CHECK: %fp64_minmax_expected = phi i64
; CHECK: %fp64_minmax_cmp = fcmp ogt double %fp64_minmax_src, %fp64_minmax_old
; CHECK: %fp64_minmax_new = select i1 %fp64_minmax_cmp, double %fp64_minmax_src, double %fp64_minmax_old
; CHECK: %fp64_minmax_cas = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %fp64_minmax_new_bits, i64 %fp64_minmax_expected
; CHECK: %fp64_minmax_ok = icmp eq i64 %fp64_minmax_cas, %fp64_minmax_expected
; CHECK: br i1 %fp64_minmax_ok, label %fp64_minmax_exit, label %fp64_minmax_loop
;
; Negative pins.
; CHECK-NOT: call double @llvm.maximumnum.f64
; CHECK-NOT: llvm.amdgcn.raw.buffer.atomic.fmax
; CHECK-NOT: atomicrmw fmax
; CHECK-NOT: atomicrmw fmaximumnum

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
