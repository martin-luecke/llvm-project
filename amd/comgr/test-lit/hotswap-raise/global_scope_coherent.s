; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=global_scope_coherent_kernel 2>/dev/null | %FileCheck %s

; A global load/store whose cpol carries device/system scope (SCOPE_DEV here)
; participates in cross-workgroup communication (decoupled-lookback scan state,
; spin-wait "ready" flags). It must lift to a *volatile* access so the AMDGPU
; backend keeps it globally coherent and the optimizer cannot hoist/eliminate it
; (otherwise a spin-wait becomes an infinite loop and a release store is dropped).
; A default-scope (CU) access in the same kernel must stay plain.
; The SCOPE_DEV store and load become volatile; the plain (CU-scope) store does
; not. (global_load_b32 lifts as an f32 load + bitcast, hence "load volatile
; float".)
; CHECK-LABEL: define amdgpu_kernel void @global_scope_coherent_kernel(
; CHECK-DAG: store volatile i32 {{.*}}, ptr addrspace(1) {{.*}}, align 4
; CHECK-DAG: load volatile float, ptr addrspace(1) {{.*}}, align 4
; CHECK-DAG: store i32 %{{[A-Za-z_][^,]*}}, ptr addrspace(1) {{.*}}, align 4
; CHECK-NOT: store volatile {{.*}}%Vgpr

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_scope_coherent_kernel
	.p2align	8
	.type	global_scope_coherent_kernel,@function
global_scope_coherent_kernel:
	s_load_b128 s[4:7], s[0:1], 0x0
	v_lshlrev_b32_e32 v2, 2, v0
	s_wait_kmcnt 0x0
	; device-scope store + load (must become volatile)
	global_store_b32 v2, v0, s[4:5] scope:SCOPE_DEV
	global_load_b32 v3, v2, s[4:5] scope:SCOPE_DEV
	s_wait_loadcnt 0x0
	; plain (default/CU scope) store -- must stay non-volatile
	global_store_b32 v2, v3, s[6:7]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_scope_coherent_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 16
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 4
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
      - { .address_space: global, .offset: 8, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .max_flat_workgroup_size: 1024
    .name:            global_scope_coherent_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:      8
    .symbol:          global_scope_coherent_kernel.kd
    .vgpr_count:      4
    .wavefront_size:  32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
