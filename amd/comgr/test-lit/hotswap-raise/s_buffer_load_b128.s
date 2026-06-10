; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --emit-ir=s_buffer_load_test_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --write-hsaco=%t.out --kernel=s_buffer_load_test_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PIPE
;
; Lift test for the scalar buffer load (`s_buffer_load_b128`). The lift
; decomposes the SGPR_128 buffer descriptor (V#): the 48-bit base is
; word0 | (word1[15:0] << 32), the immediate offset is added on, and the
; b128 access becomes four i32 loads from addrspace(1). num_records/flags
; only gate out-of-range reads (return 0), which valid kernels never rely on.

; IR-LABEL: define amdgpu_kernel void @s_buffer_load_test_kernel(
; IR: and i64 {{.*}}, 65535
; IR: [[BASE:%[^ ]+]] = or i64
; IR: [[ADDR:%[^ ]+]] = add i64 [[BASE]], 16
; IR: inttoptr i64 [[ADDR]] to ptr addrspace(1)
; IR: load i32, ptr addrspace(1)
; IR-NOT: unsupported instruction

; PIPE: raise_cli: wrote
; PIPE-SAME: s_buffer_load_test_kernel

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	s_buffer_load_test_kernel
	.p2align	8
	.type	s_buffer_load_test_kernel,@function
s_buffer_load_test_kernel:
	s_mov_b32 s8, 0
	s_mov_b32 s9, 0
	s_mov_b32 s10, 0
	s_mov_b32 s11, 0
	s_buffer_load_b128 s[12:15], s[8:11], 0x10
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, s12
	global_store_b32 v0, v1, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel s_buffer_load_test_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 16
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name: s_buffer_load_test_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 16
    .symbol: s_buffer_load_test_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
