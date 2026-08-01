; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=c5_predicate_chain_workitem_id_y_scaled_modrep_kernel 2>&1 \
; RUN:   | %FileCheck %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=c5_predicate_chain_workitem_id_y_scaled_modrep_kernel 2>&1 \
; RUN:   | %FileCheck %s

; A workitem.id.y()-derived predicate reaching a store address; raised under
; ScaledModuloReplicationProjection (auto-upgrade, also reachable via the flag).

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	c5_predicate_chain_workitem_id_y_scaled_modrep_kernel
	.type	c5_predicate_chain_workitem_id_y_scaled_modrep_kernel,@function
; CHECK: selected ScaledModuloReplicationProjection
; CHECK: define amdgpu_kernel void @c5_predicate_chain_workitem_id_y_scaled_modrep_kernel(
; CHECK-NOT: init_whole_wave
c5_predicate_chain_workitem_id_y_scaled_modrep_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	v_bfe_u32 v2, v0, 10, 10
	s_wait_kmcnt 0x0
	v_cmp_lt_u32_e64 s4, v2, 16
	v_cndmask_b32_e64 v0, -1, v0, s4
	v_mov_b32_e32 v1, v2
; The physical x/y coordinates are flattened using the scaled x extent before
; hardware lane W_s+i is remapped to the logical source thread of lane i.
; This matters for launch shapes whose x extent is narrower than a source wave.
; CHECK-DAG: %dd_group_xy{{.*}} = load i32, ptr addrspace(4) {{.+}}, align 4
; CHECK-DAG: %dd_physical_group_size_x{{.*}} = and i32 {{.+}}, 65535
; CHECK-DAG: %dd_source_group_size_x{{.*}} = udiv i32 {{.+}}, 2
; CHECK-DAG: %dd_physical_flat_tid{{.*}} = add i32 {{.+}}
; The final source wave may be partial. Its target lanes are reduced modulo the
; actual remaining population instead of being allowed to address nonexistent
; source workitems.
; CHECK-DAG: %dd_source_flat_size{{.*}} = mul i32 {{.+}}
; CHECK-DAG: %dd_source_wave_remaining{{.*}} = sub i32 {{.+}}
; CHECK-DAG: %dd_source_wave_population{{.*}} = select i1 {{.+}}
; CHECK-DAG: %dd_source_lane{{.*}} = urem i32 {{.+}}, %dd_source_wave_population
; CHECK-DAG: %dd_logical_flat_tid{{.*}} = add i32 {{.+}}
; CHECK-DAG: %dd_logical_y{{.*}} = urem i32 {{.+}}
; the x extent is advertised scaled, with a breadcrumb:
; CHECK-DAG: "amdgpu-flat-work-group-size"="1024,1024"
; CHECK-DAG: "hotswap-scaled-dispatch"="x2"
	global_store_b32 v1, v0, s[2:3]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel c5_predicate_chain_workitem_id_y_scaled_modrep_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 6
	.end_amdhsa_kernel
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:       0
        .size:         8
        .value_kind:   global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align:    8
    .kernarg_segment_size:     8
    .max_flat_workgroup_size:  512
    .name:                     c5_predicate_chain_workitem_id_y_scaled_modrep_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         c5_predicate_chain_workitem_id_y_scaled_modrep_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
