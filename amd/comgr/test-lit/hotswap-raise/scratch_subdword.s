; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=scratch_subdword_kernel 2>&1 | %FileCheck %s --check-prefix=IR

; Sub-dword FLAT scratch load/store lifted to private-segment (addrspace 5)
; accesses: byte store truncates, byte load zero-extends, short load
; sign-extends. Mirrors GLOBAL_STORE_{BYTE,SHORT} / GLOBAL_LOAD_{U,S}* but on
; private memory. Emitted by register spills (bool accumulators) in reduce
; kernels.
; IR: source_private_segment = alloca i8, i32 64, align 4, addrspace(5)
; IR: %scratch_store_trunc{{[0-9]*}} = trunc i32 {{.*}} to i8
; IR: store i8 %scratch_store_trunc{{[0-9]*}}, ptr addrspace(5) {{.*}}, align 1
; IR: %scratch_load{{[0-9]*}} = load i8, ptr addrspace(5) {{.*}}, align 1
; IR: %scratch_zext{{[0-9]*}} = zext i8 %scratch_load{{[0-9]*}} to i32
; IR: %scratch_load{{[0-9]*}} = load i16, ptr addrspace(5) {{.*}}, align 2
; IR: %scratch_sext{{[0-9]*}} = sext i16 %scratch_load{{[0-9]*}} to i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	scratch_subdword_kernel
	.p2align	8
	.type	scratch_subdword_kernel,@function
scratch_subdword_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mul_u32_u24_e32 v1, 3, v0
	scratch_store_b8 off, v1, off offset:0
	scratch_load_u8  v1, off, off offset:0
	scratch_load_i16 v2, off, off offset:4
	v_add_nc_u32_e32 v1, v1, v2
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scratch_subdword_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 64
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 2
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
		.amdhsa_enable_private_segment 1
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
    .name:           scratch_subdword_kernel
    .private_segment_fixed_size: 64
    .sgpr_count:     2
    .symbol:         scratch_subdword_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
