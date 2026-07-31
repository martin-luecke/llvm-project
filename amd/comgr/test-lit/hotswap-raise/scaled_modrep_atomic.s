; Non-idempotent atomic RMWs must issue once per source lane under a scaled
; dispatch: lane i and its active replica i+W_s both pass the emitUnderExec mask,
; so an ungated add/sub/fadd/pk_add/xor/swap would double-count the accumulator
; (the split-K / atomic-accumulate GEMM epilogue). Guards needsOneReplicaGate in
; handle-flat.cpp at both the global and flat atomic sites, which classifies by
; the idempotent exception set (and/or/min/max) and gates everything else.
; max_flat_workgroup_size=256 so the scaled block (512) fits the target max.

; The four non-idempotent RMWs -- integer add, fp add, packed fp add (all
; global), and the flat add -- each gate to replica-0 (lane_id < W_s), so there
; are four one_replica gates for four atomics.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=atom_kernel 2>&1 | %FileCheck %s
; CHECK-DAG: atomicrmw add ptr addrspace(1) {{.+}}, i32 1 monotonic
; CHECK-DAG: atomicrmw fadd ptr addrspace(1) {{.+}}, float {{.+}} monotonic
; CHECK-DAG: atomicrmw fadd ptr addrspace(1) {{.+}}, <2 x half> {{.+}} monotonic
; CHECK-DAG: atomicrmw add ptr {{.+}}, i32 1 seq_cst
; CHECK-DAG: %one_replica{{.*}} = icmp ult i32 %lane_id, 32
; CHECK-DAG: %one_replica{{.*}} = icmp ult i32 %lane_id, 32
; CHECK-DAG: %one_replica{{.*}} = icmp ult i32 %lane_id, 32
; CHECK-DAG: %one_replica{{.*}} = icmp ult i32 %lane_id, 32

; An idempotent RMW re-applies with the same operand as a no-op, so it is never
; gated -- the whole kernel is free of the replica gate.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=atom_idem_kernel 2>&1 | %FileCheck %s --check-prefix=IDEM
; IDEM-NOT: one_replica
; IDEM: atomicrmw umax
; IDEM-NOT: one_replica

; A returning atomicrmw cannot be made replica-consistent under a scaled dispatch
; -- the lane and its replica each read a different old value -- so it is refused
; regardless of idempotency: both a non-idempotent add and an idempotent max.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=atom_return_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=atom_return_max_kernel 2>&1 | %FileCheck %s --check-prefix=REFUSE
; REFUSE: returning atomic RMW under a scaled dispatch

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	atom_kernel
	.p2align	8
	.type	atom_kernel,@function
atom_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	global_atomic_add_u32 v0, v1, s[0:1]
	global_atomic_add_f32 v0, v1, s[0:1]
	global_atomic_pk_add_f16 v0, v1, s[0:1]
	flat_atomic_add_u32 v[2:3], v1
	s_endpgm

	.globl	atom_idem_kernel
	.p2align	8
	.type	atom_idem_kernel,@function
atom_idem_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	global_atomic_max_u32 v0, v1, s[0:1]
	s_endpgm

	.globl	atom_return_kernel
	.p2align	8
	.type	atom_return_kernel,@function
atom_return_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	global_atomic_add_u32 v2, v0, v1, s[0:1] th:TH_ATOMIC_RETURN
	global_store_b32 v0, v2, s[0:1]
	s_endpgm

	.globl	atom_return_max_kernel
	.p2align	8
	.type	atom_return_max_kernel,@function
atom_return_max_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 1
	global_atomic_max_u32 v2, v0, v1, s[0:1] th:TH_ATOMIC_RETURN
	global_store_b32 v0, v2, s[0:1]
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel atom_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel atom_idem_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel atom_return_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdhsa_kernel atom_return_max_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 2
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  256
    .name:                     atom_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         atom_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  256
    .name:                     atom_idem_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         atom_idem_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  256
    .name:                     atom_return_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         atom_return_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  256
    .name:                     atom_return_max_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         atom_return_max_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
