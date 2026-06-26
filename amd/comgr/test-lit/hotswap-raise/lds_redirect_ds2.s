; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:      --enable-lds-redirect --force-lds-redirect \
; RUN:      --emit-ir=lds_redirect_ds2_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; IR-shape fixture for the two-offset DS family (DS_READ2 / DS_WRITE2) under the
; LDS->global redirect.  Larger Triton kernels emit ds_store2/ds_load2 to issue
; two independent LDS accesses per instruction; the redirect must rewrite BOTH
; accesses to the per-workgroup global buffer (two GEPs through wg_lds_base),
; not just single-offset DS.  Without this, kernels using ds_*2addr_* fail to
; lift (lds-global-redirect-unsupported-variant) and fall back to untranspiled
; gfx1250 code.
;
; FORCE=1 so the fixture runs independently of a real >64 KiB kernel.

; New wg_lds_base parameter.
; CHECK-LABEL: define amdgpu_kernel void @lds_redirect_ds2_kernel(
; CHECK:       ptr addrspace(1) %wg_lds_base

; Both DS2 accesses become GEPs into the global per-WG buffer (addrspace 1),
; and the second access is offset from the first.
; CHECK: %ds2_p0 = getelementptr inbounds i8, ptr addrspace(1) %wg_lds_ptr
; CHECK: %ds2_off = add i64
; CHECK: %ds2_p1 = getelementptr inbounds i8, ptr addrspace(1) %wg_lds_ptr

; The two stores target addrspace(1), NOT LDS addrspace(3).
; CHECK: store i32 {{.*}}, ptr addrspace(1) %ds2_p0
; CHECK: store i32 {{.*}}, ptr addrspace(1) %ds2_p1

; No hardware LDS declared.
; CHECK-NOT: "amdgpu-lds-size"

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	lds_redirect_ds2_kernel
	.p2align	8
	.type	lds_redirect_ds2_kernel,@function
lds_redirect_ds2_kernel:
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_lshlrev_b32 v2, 2, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 31, v1
	ds_store_2addr_b32 v2, v0, v1 offset0:0 offset1:2
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_lshlrev_b32_e32 v1, 2, v1
	ds_load_2addr_b32 v[2:3], v1 offset0:0 offset1:4
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_redirect_ds2_kernel
		.amdhsa_group_segment_fixed_size 4096
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 4
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
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           lds_redirect_ds2_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         lds_redirect_ds2_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
