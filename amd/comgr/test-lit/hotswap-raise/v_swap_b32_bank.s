; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=v_swap_b32_bank_kernel 2>/dev/null | %FileCheck %s

; v_swap_b32 under s_set_vgpr_msb bank select lowered with gfx125x high-VGPR (VGPR MSB) addressing.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	v_swap_b32_bank_kernel
	.p2align	8
	.type	v_swap_b32_bank_kernel,@function
; CHECK-LABEL: define amdgpu_kernel void @v_swap_b32_bank_kernel(
v_swap_b32_bank_kernel:
	global_load_b32 v1, v0, s[0:1]
	global_load_b32 v2, v0, s[2:3]
	s_set_vgpr_msb 0x40
	v_swap_b32 v1, v2
	; Cross-widening (wave32->wave64) hardens global loads against back-end
	; if-conversion by nesting them in a `memop_do`/`memop_cont` diamond inside
	; the source-EXEC `spe_do` guard (see emitMemOpUnderExecHardened). The v2
	; load result therefore lands in the memop_cont phi and propagates out
	; through the spe_skip phi into the v_swap result.
	; CHECK: %[[V2PROP:Vgpr2\.[0-9]+]] = phi i32 [ %[[V2LOAD:Vgpr2\.[0-9]+]], %memop_cont{{[0-9]+}} ]
	; CHECK: %[[V2LOAD]] = phi i32 [ %{{[0-9]+}}, %memop_do{{[0-9]+}} ]
	; CHECK: %[[V2SWAP:Vgpr2\.[0-9]+]] = phi i32 {{.*}}[ %[[V2PROP]], %spe_skip{{[0-9]+}} ]
	s_set_vgpr_msb 0
	global_store_b32 v0, v1, s[0:1]
	global_store_b32 v0, v2, s[2:3]
	; CHECK: store i32 %[[V2SWAP]],
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_swap_b32_bank_kernel
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 8
		.amdhsa_wavefront_size32 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_swap_b32_bank_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_swap_b32_bank_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
