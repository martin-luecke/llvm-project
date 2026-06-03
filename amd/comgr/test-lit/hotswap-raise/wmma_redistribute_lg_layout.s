; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=wmma_redistribute_lg_layout_kernel 2>&1 \
; RUN:   | %FileCheck %s
;
; Regression guard for the WMMA->MFMA input redistribution LG assignment.
; MI400 Shader Programming Guide section 4.6.12.2 canonical layout:
;
;   LG 0 (lanes 0-15)  reads (lo half, VGPR-pair G)   -> K=0..3
;   LG 1 (lanes 16-31) reads (lo half, VGPR-pair G+2) -> K=4..7
;   LG 2 (lanes 32-47) reads (hi half, VGPR-pair G)   -> K=8..11
;   LG 3 (lanes 48-63) reads (hi half, VGPR-pair G+2) -> K=12..15
;
; redistributeInput in wmma-lowering.cpp produces four bpermute calls per
; G with address order AddrLo, AddrLo, AddrHi, AddrHi. A regression that
; reintroduced the wrong order would interleave AddrLo and AddrHi, failing
; the CHECK-NEXT sequence below. The kernel uses v_wmma_f32_16x16x32_bf16
; (K=32 BF16) under --enable-wave-native.

; CHECK-LABEL: define amdgpu_kernel void @wmma_redistribute_lg_layout_kernel(
; CHECK: %addr_lo = shl i32 %{{.+}}, 2
; CHECK: %addr_hi = shl i32 %{{.+}}, 2
; CHECK: %lane_grp = lshr i32 %{{.+}}, 4
; LG0 and LG1 both read from addr_lo; LG2 and LG3 from addr_hi.
; CHECK: %[[V0:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_lo, i32 %[[DWG:dw[0-9]*]])
; CHECK-NEXT: %[[V1:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_lo, i32 %[[DWG2:dw[0-9]*]])
; CHECK-NEXT: %[[V2:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_hi, i32 %[[DWG]])
; CHECK-NEXT: %[[V3:bperm[0-9]*]] = call i32 @llvm.amdgcn.ds.bpermute(i32 %addr_hi, i32 %[[DWG2]])
; selectByLaneGroup mux selects by lane_grp comparison.
; CHECK: %[[EQ2:[0-9]+]] = icmp eq i32 %lane_grp, 2
; CHECK-NEXT: %{{[0-9]+}} = select i1 %[[EQ2]], i32 %[[V2]], i32 %[[V3]]
; CHECK: %[[EQ1:[0-9]+]] = icmp eq i32 %lane_grp, 1
; CHECK-NEXT: %{{[0-9]+}} = select i1 %[[EQ1]], i32 %[[V1]], i32 %{{[0-9]+}}
; CHECK: %[[EQ0:[0-9]+]] = icmp eq i32 %lane_grp, 0
; CHECK-NEXT: %{{[0-9]+}} = select i1 %[[EQ0]], i32 %[[V0]], i32 %{{[0-9]+}}


	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	wmma_redistribute_lg_layout_kernel
	.p2align	8
	.type	wmma_redistribute_lg_layout_kernel,@function
wmma_redistribute_lg_layout_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_clause 0x1
	s_load_b128 s[24:27], s[0:1], 0x0
	s_load_b64 s[28:29], s[0:1], 0x10
	v_mov_b32_e32 v24, 0
	s_wait_kmcnt 0x0
	s_load_b256 s[0:7], s[24:25], 0x0
	s_load_b256 s[8:15], s[26:27], 0x0
	s_load_b256 s[16:23], s[28:29], 0x0
	s_wait_kmcnt 0x0
	v_mov_b64_e32 v[0:1], s[0:1]
	v_mov_b64_e32 v[8:9], s[8:9]
	v_mov_b64_e32 v[16:17], s[16:17]
	v_mov_b64_e32 v[2:3], s[2:3]
	v_mov_b64_e32 v[4:5], s[4:5]
	v_mov_b64_e32 v[6:7], s[6:7]
	v_mov_b64_e32 v[10:11], s[10:11]
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b64_e32 v[14:15], s[14:15]
	v_mov_b64_e32 v[18:19], s[18:19]
	v_mov_b64_e32 v[20:21], s[20:21]
	v_mov_b64_e32 v[22:23], s[22:23]
	s_delay_alu instid0(VALU_DEP_1)
	v_wmma_f32_16x16x32_bf16 v[16:23], v[0:7], v[8:15], v[16:23]
	s_clause 0x1
	global_store_b128 v24, v[20:23], s[28:29] offset:16
	global_store_b128 v24, v[16:19], s[28:29]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel wmma_redistribute_lg_layout_kernel
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 25
		.amdhsa_next_free_sgpr 30
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 2
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
      - { .address_space:  global, .offset:         8, .size:           8, .value_kind:     global_buffer }
      - { .address_space:  global, .offset:         16, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           wmma_redistribute_lg_layout_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     30
    .symbol:         wmma_redistribute_lg_layout_kernel.kd
    .vgpr_count:     25
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
