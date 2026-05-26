; RUN: %llvm_mc -mcpu=gfx942 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --emit-ir 2>/dev/null | %FileCheck %s
;
; RTN-form companion to ds_add_f64.s. The `_RTN` is an infix in DS
; pseudo names (DS_ADD_RTN_F64, not DS_ADD_F64_RTN), so the standard
; trailing-suffix strip rule does not fire. The opcode-map declares
; an explicit DS_ADD_RTN_F64 -> DS_ADD_F64 alias and the handler
; routes the writeback through `di.numDefs > 0`.
;
; Pins both halves of the contract:
;   (1) atomicrmw fadd double in addrspace(3) -- same lift as non-RTN
;   (2) the returned pre-add value is written back to a VGPR pair

; CHECK-LABEL: define amdgpu_kernel void @ds_add_rtn_f64_kernel(
; CHECK: %[[RMW:.+]] = atomicrmw fadd ptr addrspace(3) %{{[^,]+}}, double %{{[^ ]+}}
; CHECK: bitcast double %[[RMW]] to i64
; CHECK-NOT: atomicrmw fadd float

	.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
	.amdhsa_code_object_version 6
	.text
	.globl	ds_add_rtn_f64_kernel
	.p2align	8
	.type	ds_add_rtn_f64_kernel,@function
ds_add_rtn_f64_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v2, 0
	v_mov_b32_e32 v3, 0x3ff00000
	;;#ASMSTART
	ds_add_rtn_f64 v[4:5], v0, v[2:3]
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ds_add_rtn_f64_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_group_segment_fixed_size 16
		.amdhsa_next_free_vgpr 6
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
    .name:           ds_add_rtn_f64_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         ds_add_rtn_f64_kernel.kd
    .vgpr_count:     6
    .wavefront_size: 64
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
