; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>/dev/null | %FileCheck %s
;
; Lift test for ds_add_f64 (and the same pseudo's `ds_add_rtn_f64`
; alias which collapses via the `_RTN` suffix rule). LDS 64-bit FP
; add, native on gfx90a/gfx940-family. The raiser emits an addrspace(3)
; `atomicrmw fadd double`; AtomicExpandPass re-emits the native opcode
; on same-target re-codegen or expands to a ds_cmpst_b64 CAS loop on
; subtargets that don't support it.

; CHECK-LABEL: define amdgpu_kernel void @ds_add_f64_kernel(
; CHECK: atomicrmw fadd ptr addrspace(3) %{{[^,]+}}, double %{{[^ ]+}}
; CHECK-NOT: atomicrmw fadd float
; CHECK-NOT: atomicrmw fadd ptr addrspace(0)
; CHECK-NOT: atomicrmw fadd ptr addrspace(1)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_add_f64_kernel
	.p2align	8
	.type	ds_add_f64_kernel,@function
ds_add_f64_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0x3ff00000
	;;#ASMSTART
	ds_add_f64 v0, v[2:3]
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_add_f64_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_group_segment_fixed_size 16
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
  - .args: []
    .group_segment_fixed_size: 16
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           ds_add_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_add_f64_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
