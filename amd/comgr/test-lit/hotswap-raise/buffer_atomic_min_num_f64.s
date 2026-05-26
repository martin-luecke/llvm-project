; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=buffer_atomic_min_num_f64_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; CAS-loop lift test for buffer_atomic_min_num_f64 from a gfx1250
; source binary. The disassembler maps both `buffer_atomic_min_f64`
; (gfx942) and `buffer_atomic_min_num_f64` (gfx1250) to the same
; LLVM pseudo `BUFFER_ATOMIC_MIN_F64`, but the HW semantics differ:
; the `_num_` variant is IEEE 754-2019 minimumNumber (sp3 manual
; sec 8.7.23) while the legacy gfx942 op is a raw `<` comparator.
;
; The raiser disambiguates by checking the assembled mnemonic for
; `_num_`; this fixture pins the IEEE 754-2019 path where the loop
; body uses `llvm.minimumnum.f64` instead of the raw `fcmp olt`
; + `select` used for gfx942. The CAS loop structure itself
; (raw_buffer_atomic_cmpswap.i64 + retry) is shared between the
; two paths; the only difference is the per-iteration comparator.
;
; Sibling fixture buffer_atomic_min_f64.s pins the gfx942 raw path.

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_min_num_f64_kernel(
; CHECK: %fp64_minmax_init = call i64 @llvm.amdgcn.raw.buffer.load.i64
; CHECK: fp64_minmax_loop:
; CHECK: %fp64_minmax_expected = phi i64
; IEEE 754-2019 minimumNumber semantics (NOT `fcmp olt` + `select`,
; that's the gfx942 path):
; CHECK: %fp64_minmax_new = call double @llvm.minimumnum.f64(double %fp64_minmax_old, double %fp64_minmax_src)
; CHECK: %fp64_minmax_cas = call i64 @llvm.amdgcn.raw.buffer.atomic.cmpswap.i64(i64 %fp64_minmax_new_bits, i64 %fp64_minmax_expected
; CHECK: %fp64_minmax_ok = icmp eq i64 %fp64_minmax_cas, %fp64_minmax_expected
; CHECK: br i1 %fp64_minmax_ok, label %fp64_minmax_exit, label %fp64_minmax_loop
;
; Negative pins: gfx942 raw path must not leak in.
; CHECK-NOT: fcmp olt
; CHECK-NOT: llvm.amdgcn.raw.buffer.atomic.fmin
; CHECK-NOT: atomicrmw fmin

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_min_num_f64_kernel
	.p2align	8
	.type	buffer_atomic_min_num_f64_kernel,@function
buffer_atomic_min_num_f64_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b96 s[0:2], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 3, v0
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s2
	v_mov_b32_e32 v3, 0x3ff00000
	s_mov_b32 s2, -1
	;;#ASMSTART
	buffer_atomic_min_num_f64 v[2:3], v0, s[0:3], null offen
	s_wait_loadcnt 0

	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_min_num_f64_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
      - { .offset:         8, .size:           4, .value_kind:     by_value }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_min_num_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_atomic_min_num_f64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
