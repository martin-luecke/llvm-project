; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --emit-ir=global_store_async_from_lds_b128_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
; RUN: %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx1250 \
; RUN:   --write-hsaco=%t.out --kernel=global_store_async_from_lds_b128_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=PIPE
; RUN: %llvm-objdump -d %t.out | %FileCheck %s --check-prefix=DISASM
;
; Lift test for the gfx1250 async global store-from-LDS
; (`global_store_async_from_lds_b128`, SADDR form). Each lane copies 16 bytes
; from a per-lane LDS source (v1 = LDS base offset) to saddr + vaddr. Same-target
; gfx1250 -> gfx1250 emits the native intrinsic and the backend re-selects the
; identical instruction (round-trip identity).

; IR-LABEL: define amdgpu_kernel void @global_store_async_from_lds_b128_kernel(
; IR: call void @llvm.amdgcn.global.store.async.from.lds.b128(ptr addrspace(1) {{[^,]+}}, ptr addrspace(3) {{[^,]+}}, i32 {{[^,]+}}, i32 {{[^)]+}})
; IR-NOT: unsupported instruction

; PIPE: raise_cli: wrote
; PIPE-SAME: global_store_async_from_lds_b128_kernel

; DISASM: global_store_async_from_lds_b128

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_store_async_from_lds_b128_kernel
	.p2align	8
	.type	global_store_async_from_lds_b128_kernel,@function
global_store_async_from_lds_b128_kernel:
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, 0
	;;#ASMSTART
	global_store_async_from_lds_b128 v0, v1, s[0:1]
	;;#ASMEND
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_store_async_from_lds_b128_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 2
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
    .name: global_store_async_from_lds_b128_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 2
    .symbol: global_store_async_from_lds_b128_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
