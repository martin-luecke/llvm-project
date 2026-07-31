; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=buffer_store_short_sentinel_srd_kernel,buffer_store_short_finite_srd_kernel,buffer_store_short_allones_srd_kernel,buffer_store_short_ambiguous_srd_kernel 2>/dev/null | %FileCheck %s --check-prefixes=CHECK,CDNA
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 --emit-ir=buffer_store_short_sentinel_srd_kernel,buffer_store_short_finite_srd_kernel,buffer_store_short_allones_srd_kernel,buffer_store_short_ambiguous_srd_kernel 2>/dev/null | %FileCheck %s --check-prefixes=CHECK,RDNA

; Buffer store b16 SRD num-records reconstruction and target format word.
;
; A gfx1250 source V# splits NUM_RECORDS across dword1[31:25] (low 7 bits),
; dword2 (bits [38:7]) and dword3[5:0] (bits [44:39]), so dword2 alone is the
; byte extent >> 7. The <4 x i32> descriptor and the addrspace(8) resource must
; decode it identically, or the two forms disagree about the same buffer; the
; kernels below vary dword2 over the sentinel / finite / all-ones / ambiguous
; shapes. Word3 is the target's raw-buffer format (gfx942: 0x27000 == 159744),
; never the bare DATA_FORMAT_32 (131072).

; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_sentinel_srd_kernel(
; CHECK: and i32 %{{.+}}, 127
; CHECK: zext i32 16777215 to i64
; CHECK: shl i64 %{{.+}}, 7
; CHECK: shl i64 %{{.+}}, 39
; CHECK: %[[FULL:mubuf_raw_num_records_full[0-9]*]] = or i64
; CHECK: %[[NR:.+]] = select i1 %{{.+}}, i64 2147483646, i64 %[[FULL]]
; CHECK: %[[W2:.+]] = trunc i64 %[[NR]] to i32
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; CDNA: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; RDNA: insertelement <4 x i32> {{.*}}, i32 822173696, i64 3
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(

; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_finite_srd_kernel(
; CHECK: zext i32 4096 to i64
; CHECK: shl i64 %{{.+}}, 7
; CHECK: %[[FULLF:mubuf_raw_num_records_full[0-9]*]] = or i64
; CHECK: %[[NRF:.+]] = select i1 %{{.+}}, i64 2147483646, i64 %[[FULLF]]
; CHECK: %[[W2F:.+]] = trunc i64 %[[NRF]] to i32
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; CDNA: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; RDNA: insertelement <4 x i32> {{.*}}, i32 822173696, i64 3
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(

; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_allones_srd_kernel(
; CHECK: zext i32 -1 to i64
; CHECK: shl i64 %{{.+}}, 7
; CHECK: %[[FULLA:mubuf_raw_num_records_full[0-9]*]] = or i64
; CHECK: select i1 %{{.+}}, i64 2147483646, i64 %[[FULLA]]
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 131072, i64 3
; CDNA: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; RDNA: insertelement <4 x i32> {{.*}}, i32 822173696, i64 3
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(

; CHECK-LABEL: define amdgpu_kernel void @buffer_store_short_ambiguous_srd_kernel(
; CHECK: zext i32 16777215 to i64
; CHECK: and i32 1, 63
; CHECK: shl i64 %{{.+}}, 39
; CHECK: %[[FULLB:mubuf_raw_num_records_full[0-9]*]] = or i64
; CHECK: select i1 %{{.+}}, i64 2147483646, i64 %[[FULLB]]
; CHECK: call void @llvm.amdgcn.raw.buffer.store.i16(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text

	.globl	buffer_store_short_sentinel_srd_kernel
	.p2align	8
	.type	buffer_store_short_sentinel_srd_kernel,@function
buffer_store_short_sentinel_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_finite_srd_kernel
	.p2align	8
	.type	buffer_store_short_finite_srd_kernel,@function
buffer_store_short_finite_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 0
	s_mov_b32 s2, 4096
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_allones_srd_kernel
	.p2align	8
	.type	buffer_store_short_allones_srd_kernel,@function
buffer_store_short_allones_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 0
	s_mov_b32 s2, -1
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.globl	buffer_store_short_ambiguous_srd_kernel
	.p2align	8
	.type	buffer_store_short_ambiguous_srd_kernel,@function
buffer_store_short_ambiguous_srd_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 1, v0
	v_mov_b32_e32 v1, 0x1234
	s_mov_b32 s3, 1
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_store_b16 v1, v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_short_sentinel_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_finite_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_allones_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel

	.amdhsa_kernel buffer_store_short_ambiguous_srd_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_sentinel_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_sentinel_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_finite_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_finite_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_allones_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_allones_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args:
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           buffer_store_short_ambiguous_srd_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_short_ambiguous_srd_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
