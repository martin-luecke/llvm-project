; REQUIRES: comgr-has-hotswap-transpile
; RUN: %llvm-mc -triple=amdgpu9.50-amd-amdhsa -filetype=obj %s -o %t.o
; RUN: %ld.lld -shared %t.o -o %t.hsaco
; RUN: %hotswap_transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_loads | %FileCheck %s --check-prefix=IR
; RUN: not %hotswap_transpile_cli %t.hsaco --target-isa=gfx950 \
; RUN:   --emit-ir=ds_agpr 2>&1 | %FileCheck %s --check-prefix=REFUSE

	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.text

	.globl ds_loads
	.p2align 8
	.type ds_loads,@function
; IR-LABEL: define amdgpu_kernel void @ds_loads(
ds_loads:
	s_load_dwordx2 s[2:3], s[0:1], 0
	s_waitcnt lgkmcnt(0)
	v_mov_b32 v20, s2
	v_mov_b32 v21, s3
	v_mov_b32 v0, 0x10000
; IR: [[ADDR32:%.+]] = add i32 65536, 0
; IR: [[PTR32:%.+]] = inttoptr i32 [[ADDR32]] to ptr addrspace(3)
; IR-NEXT: [[LOAD32:%.+]] = load i32, ptr addrspace(3) [[PTR32]], align 1
; IR: [[DEST32_0:%.+]] = phi i32 [ [[LOAD32]], %{{.+}} ], [ undef, %{{.+}} ]
	ds_read_b32 v1, v0
; IR: [[ADDR64:%.+]] = add i32 65536, 8
; IR: [[PTR64:%.+]] = inttoptr i32 [[ADDR64]] to ptr addrspace(3)
; IR-NEXT: [[LOAD64:%.+]] = load i64, ptr addrspace(3) [[PTR64]], align 1
; IR-NEXT: [[WORD64_0:%.+]] = trunc i64 [[LOAD64]] to i32
; IR-NEXT: [[SHIFT64_1:%.+]] = lshr i64 [[LOAD64]], 32
; IR-NEXT: [[WORD64_1:%.+]] = trunc i64 [[SHIFT64_1]] to i32
; IR: [[DEST64_0:%.+]] = phi i32 [ [[WORD64_0]], %{{.+}} ], [ undef, %{{.+}} ]
; IR: [[DEST64_1:%.+]] = phi i32 [ [[WORD64_1]], %{{.+}} ], [ undef, %{{.+}} ]
	ds_read_b64 v[2:3], v0 offset:8
; IR: [[ADDR128:%.+]] = add i32 65536, 16
; IR: [[PTR128:%.+]] = inttoptr i32 [[ADDR128]] to ptr addrspace(3)
; IR-NEXT: [[LOAD128:%.+]] = load <4 x i32>, ptr addrspace(3) [[PTR128]], align 1
; IR-NEXT: [[BITS128:%.+]] = bitcast <4 x i32> [[LOAD128]] to i128
; IR-NEXT: [[WORD128_0:%.+]] = trunc i128 [[BITS128]] to i32
; IR-NEXT: [[SHIFT128_1:%.+]] = lshr i128 [[BITS128]], 32
; IR-NEXT: [[WORD128_1:%.+]] = trunc i128 [[SHIFT128_1]] to i32
; IR-NEXT: [[SHIFT128_2:%.+]] = lshr i128 [[BITS128]], 64
; IR-NEXT: [[WORD128_2:%.+]] = trunc i128 [[SHIFT128_2]] to i32
; IR-NEXT: [[SHIFT128_3:%.+]] = lshr i128 [[BITS128]], 96
; IR-NEXT: [[WORD128_3:%.+]] = trunc i128 [[SHIFT128_3]] to i32
; IR: [[DEST128_0:%.+]] = phi i32 [ [[WORD128_0]], %{{.+}} ], [ undef, %{{.+}} ]
; IR: [[DEST128_1:%.+]] = phi i32 [ [[WORD128_1]], %{{.+}} ], [ undef, %{{.+}} ]
; IR: [[DEST128_2:%.+]] = phi i32 [ [[WORD128_2]], %{{.+}} ], [ undef, %{{.+}} ]
; IR: [[DEST128_3:%.+]] = phi i32 [ [[WORD128_3]], %{{.+}} ], [ undef, %{{.+}} ]
	ds_read_b128 v[4:7], v0 offset:16
	s_waitcnt lgkmcnt(0)
; IR: store i32 [[DEST32_0]], ptr addrspace(1)
	global_store_dword v[20:21], v1, off offset:0
; IR: store i32 [[DEST64_0]], ptr addrspace(1)
	global_store_dword v[20:21], v2, off offset:4
; IR: store i32 [[DEST64_1]], ptr addrspace(1)
	global_store_dword v[20:21], v3, off offset:8
; IR: store i32 [[DEST128_0]], ptr addrspace(1)
	global_store_dword v[20:21], v4, off offset:12
; IR: store i32 [[DEST128_1]], ptr addrspace(1)
	global_store_dword v[20:21], v5, off offset:16
; IR: store i32 [[DEST128_2]], ptr addrspace(1)
	global_store_dword v[20:21], v6, off offset:20
; IR: store i32 [[DEST128_3]], ptr addrspace(1)
	global_store_dword v[20:21], v7, off offset:24
	s_endpgm

	.globl ds_agpr
	.p2align 8
	.type ds_agpr,@function
ds_agpr:
; REFUSE: unsupported-instruction-form: ds_read_b32 [DS]
; REFUSE-SAME: in kernel 'ds_agpr'
; REFUSE-SAME: DS loads require a VGPR destination
	ds_read_b32 a0, v0
	s_endpgm

	.section .rodata,"a",@progbits
	.p2align 6
	.amdhsa_kernel ds_loads
		.amdhsa_group_segment_fixed_size 65568
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 24
	.end_amdhsa_kernel
	.amdhsa_kernel ds_agpr
		.amdhsa_group_segment_fixed_size 256
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 4
		.amdhsa_accum_offset 24
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .name: ds_loads
    .symbol: ds_loads.kd
    .group_segment_fixed_size: 65568
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 24
    .wavefront_size: 64
  - .name: ds_agpr
    .symbol: ds_agpr.kd
    .group_segment_fixed_size: 256
    .kernarg_segment_size: 8
    .kernarg_segment_align: 8
    .private_segment_fixed_size: 0
    .max_flat_workgroup_size: 64
    .sgpr_count: 4
    .vgpr_count: 24
    .wavefront_size: 64
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
