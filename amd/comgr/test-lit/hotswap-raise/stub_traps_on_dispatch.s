; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && env HSA_HOTSWAP_STUB_FAILED_KERNELS=1 %raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --kernel=stub_trap_kernel --write-hsaco=%t.co \
; RUN:   && %llvm-objdump -d %t.co | %FileCheck %s

; An untranslatable kernel is replaced by a trapping stub. Dispatching it is a
; hard device fault -- the stub must lower to `s_trap`, not a silent `s_endpgm`.
; CHECK-LABEL: <stub_trap_kernel>:
; CHECK: s_trap
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	stub_trap_kernel
	.p2align	8
	.type	stub_trap_kernel,@function
stub_trap_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	s_flbit_i32_b32 exec_lo, 0x12345678
	s_mov_b64 exec, -1
	s_wait_kmcnt 0x0
	global_store_b32 v0, v0, s[0:1]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel stub_trap_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 2
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
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 1024
    .name:           stub_trap_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         stub_trap_kernel.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
