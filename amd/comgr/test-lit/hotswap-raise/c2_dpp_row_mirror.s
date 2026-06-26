; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c2_dpp_row_mirror_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Positive canary for the DPP16 row_mirror mode under cross-widening
; (gfx1250 wave32 -> gfx942 wave64).  row_mirror reverses the lane
; index within each 16-lane row:
;
;   lane[n].src0 = lane[(n & 0x30) + (15 - (n & 0xf))].src0
;
; The subtraction is confined to the low 4 bits, so the source lane
; remains in the same 16-lane row.  The rewrite uses `15 - withinRow`,
; and InRange is unconditionally true (the mirror stays in-row).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	c2_dpp_row_mirror_kernel
	.p2align	8
	.type	c2_dpp_row_mirror_kernel,@function
c2_dpp_row_mirror_kernel:
; CHECK-LABEL: define amdgpu_kernel void @c2_dpp_row_mirror_kernel(
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
	v_mov_b32_dpp v0, v0 row_mirror row_mask:0xf bank_mask:0xf
; CHECK: and i32 %{{.+}}, 15
; CHECK: sub i32 15, %{{.+}}
; CHECK: select i1 true, i32 %{{.+}}, i32 0
; CHECK: call i32 @llvm.amdgcn.ds.bpermute(i32 %{{[^,]+}}, i32 %{{[^,]+}})
; CHECK-NOT: call i32 @llvm.amdgcn.update.dpp.i32(
	s_endpgm
; CHECK: declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c2_dpp_row_mirror_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:           []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           c2_dpp_row_mirror_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     0
    .symbol:         c2_dpp_row_mirror_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
