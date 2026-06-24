; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 --emit-ir=wait_storecnt_dscnt_kernel 2>/dev/null | %FileCheck %s
;
; gfx12 splits the unified `s_barrier` into `s_barrier_signal` /
; `s_barrier_wait`, and uses combined wait counters to drain in-flight
; memory before the rendezvous.  Triton's prefill `_fwd_kernel` stages Q/K
; into LDS and emits
;
;     ds_store_b128 ...
;     s_wait_storecnt_dscnt 0x0      ; drain LDS stores before the barrier
;     s_barrier_signal -1
;     s_barrier_wait 0xffff
;     ds_load_b128 ...               ; read what *other* waves stored
;
; On gfx11 the LDS stores are async (tracked by lgkmcnt) and `s_barrier`
; does NOT drain them, so the explicit wait MUST survive translation as an
; `s_waitcnt lgkmcnt(0)` *before* the barrier.  `s_wait_storecnt_dscnt` was
; not in the opcode map, so it lifted to CanonicalOp::Unknown and was
; dropped by the generic SOPP no-op arm -- the barrier then rendezvoused
; with stores still in flight, a cross-wave race that single-wave
; SIInsertWaitcnts cannot recover, producing non-deterministic NaN output.
;
; The wait must lower to `llvm.amdgcn.s.waitcnt` and appear BEFORE the
; barrier call.

; CHECK-LABEL: define amdgpu_kernel void @wait_storecnt_dscnt_kernel(
; CHECK: store i32 {{.*}}, ptr addrspace(3)
; CHECK: call void @llvm.amdgcn.s.waitcnt(i32 0)
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: load i32, ptr addrspace(3)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	wait_storecnt_dscnt_kernel
	.p2align	8
	.type	wait_storecnt_dscnt_kernel,@function
wait_storecnt_dscnt_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	v_mov_b32_e32 v2, 0x42
	ds_store_b32 v1, v2
	s_wait_storecnt_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait 0xffff
	ds_load_b32 v3, v1
	s_wait_dscnt 0x0
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b32 v3, v1, s[0:3], null offen
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wait_storecnt_dscnt_kernel
		.amdhsa_group_segment_fixed_size 1024
		.amdhsa_kernarg_size 8
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
    .group_segment_fixed_size: 1024
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           wait_storecnt_dscnt_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         wait_storecnt_dscnt_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
