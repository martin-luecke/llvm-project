; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=regfile_seed_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; Regression for the VGPR/SGPR/AGPR entry-block seeding in AllocaRegFile::init.
;
; Under SPE per-op predication a register written by the source is modeled as a
; conditional store (spe_do), so its SSA phi has a skip edge carrying the
; register's prior value. Without an entry-block init store that prior value is
; `undef` -- undefined behaviour that the AMDGPU backend materialises to
; whatever the register allocator leaves in the physical register. That value
; is allocation-dependent, so byte-identical lifted IR silently computes
; different results on backends that allocate differently (the gfx950 vs gfx942
; wave64 divergence: gfx950's v_bitop3 fusion perturbs allocation and a stale
; softmax value lands where a byte offset is expected). Seeding the register
; file makes the skip edge a deterministic 0.
;
; CHECK-LABEL: define amdgpu_kernel void @regfile_seed_kernel(
; The predicated write of v0 (constant 7) reads back 0 on the skip edge, not
; undef, and that phi value is what feeds the store.
; CHECK: %[[SEED:Vgpr[0-9.]+]] = phi i32 [ 7, {{[^ ]+}} ], [ 0, {{[^ ]+}} ]
; CHECK: store i32 %[[SEED]], ptr addrspace(1)
; No modeled-register read may resolve to undef/poison.
; CHECK-NOT: phi i32 [ {{.*}}undef

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	regfile_seed_kernel
	.p2align	8
	.type	regfile_seed_kernel,@function
regfile_seed_kernel:
	s_load_b128 s[0:3], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v5, 7
	v_mov_b64_e64 v[2:3], s[0:1]
	global_store_b32 v[2:3], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel regfile_seed_kernel
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:           regfile_seed_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         regfile_seed_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
