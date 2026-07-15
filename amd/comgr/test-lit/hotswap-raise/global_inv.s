; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=global_inv_kernel \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -defsym SCOPE_SYS=1 -o %t.sys.o \
; RUN:   && %ld_lld -shared %t.sys.o -o %t.sys.hsaco \
; RUN:   && %raise_cli %t.sys.hsaco --target-isa=gfx942 --emit-ir=global_inv_kernel \
; RUN:   | %FileCheck %s --check-prefix=IR-SYS
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -defsym SCOPE_CU=1 -o %t.cu.o \
; RUN:   && %ld_lld -shared %t.cu.o -o %t.cu.hsaco \
; RUN:   && %raise_cli %t.cu.hsaco --target-isa=gfx942 --emit-ir=global_inv_kernel \
; RUN:   | %FileCheck %s --check-prefix=IR-CU
; RUN: %llvm_mc -mcpu=gfx1250 %s -defsym SCOPE_SE=1 -o %t.se.o \
; RUN:   && %ld_lld -shared %t.se.o -o %t.se.hsaco \
; RUN:   && %not %raise_cli %t.se.hsaco --target-isa=gfx942 --emit-ir=global_inv_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE-SE

; global_inv scope lowering: SCOPE_DEV/SYS -> acquire fence; SCOPE_CU no-op;
; SCOPE_SE refused. Mirrors global_wb (the release/writeback counterpart).
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_inv_kernel
	.p2align	8
	.type	global_inv_kernel,@function
global_inv_kernel:
	.ifdef SCOPE_SYS
; IR-SYS-LABEL: define amdgpu_kernel void @global_inv_kernel(
; IR-SYS: fence acquire
	global_inv scope:SCOPE_SYS
	.else
	.ifdef SCOPE_CU
; IR-CU-LABEL: define amdgpu_kernel void @global_inv_kernel(
; IR-CU-NOT: fence
	global_inv
	.else
	.ifdef SCOPE_SE
; REFUSE-SE: failed to raise: unsupported-instruction-form: global_inv [FLAT]
; REFUSE-SE: SCOPE_SE cannot be represented
	global_inv scope:SCOPE_SE
	.else
; IR-LABEL: define amdgpu_kernel void @global_inv_kernel(
; IR: fence syncscope("agent") acquire
	global_inv scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	.endif
	.endif
	.endif
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_inv_kernel
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 0
		.amdhsa_next_free_sgpr 0
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name: global_inv_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 0
    .symbol: global_inv_kernel.kd
    .vgpr_count: 0
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
