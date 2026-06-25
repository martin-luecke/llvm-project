; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=cfg_walker_fallthrough_stub_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Wave32 -> wave64 (WaveNative) coverage for the raiser CFG walker's waterfall
; fall-through stub handling (see raiser.cpp): the stub after the loop must
; raise, emitting its compare+select.
;
; The property under test is *structural*, not just "the fcmp exists
; somewhere": the stub's compare+select must be emitted INSIDE the loop's
; fall-through successor block, ahead of that block's own terminator.  The bug
; this guards against switched the IRBuilder to a block leader with
; SetInsertPoint(BB) (== BB->end()); when a handler had already terminated that
; leader, the stub body landed *after* the terminator and was then either
; dropped by the walker's "terminated block + non-leader offset" guard or left
; stranding the terminator mid-block.  The CHECK-NOT below fails if any
; terminator precedes the stub body.
;
; NOTE: against the current parent this passes both with and without the
; SetInsertPoint-before-terminator fix -- in a release build the source-offset
; leader blocks are still empty when the walker reaches them, so the fixed and
; old code pick the same insertion point and emit identical IR.  The check is a
; forward regression guard on the block shape the fix establishes, not a
; differential reproduction (which needs a handler that fills+terminates a
; forward source block -- not expressible in a minimal kernel).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	cfg_walker_fallthrough_stub_kernel
	.p2align	8
	.type	cfg_walker_fallthrough_stub_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @cfg_walker_fallthrough_stub_kernel(
;
; The loop body stores through a divergent address, then ends with a
; conditional branch: back-edge to the loop header, fall-through to the stub.
; CHECK:      call void @llvm.amdgcn.raw.buffer.store
; CHECK:      br i1 %{{[a-zA-Z0-9_.]+}}, label %[[LOOP:bb_0x[0-9A-Fa-f]+]], label %[[STUB:bb_0x[0-9A-Fa-f]+]]
; The stub is its own block; its compare+select must appear in-block, ahead of
; the terminator.  No terminator may precede the fcmp inside the stub block.
; CHECK:      [[STUB]]:
; CHECK-NOT:  br i1
; CHECK-NOT:  br label
; CHECK:      fcmp ord float
; CHECK-NEXT: select i1
; CHECK:      br i1
cfg_walker_fallthrough_stub_kernel:
	v_mbcnt_lo_u32_b32 v0, -1, 0
	v_mov_b32_e32 v1, 0x3f800000
	s_mov_b32 s2, exec_lo
	s_mov_b32 s3, exec_hi
	v_mov_b32_e32 v3, s4
.LBB0_wf1:
	v_readfirstlane_b32 s8, v3
	v_cmp_eq_u32_e32 vcc_lo, s8, v3
	s_and_saveexec_b32 s8, vcc_lo
	buffer_store_dword v0, v0, s[4:7], null offen
	s_xor_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_wf1
; The fall-through stub's compare+select must reach the lifted IR, in-block.
	s_mov_b32 exec_lo, s2
	s_mov_b32 exec_hi, s3
	v_cmp_o_f32_e32 vcc_lo, v1, v1
	v_cndmask_b32_e32 v1, 0, v1, vcc_lo
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel cfg_walker_fallthrough_stub_kernel
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 10
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 64
    .name:           cfg_walker_fallthrough_stub_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .symbol:         cfg_walker_fallthrough_stub_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
