; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:      --enable-lds-redirect \
; RUN:      --emit-ir=lds_redirect_dynamic_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; IR-shape fixture for the *dynamic*-LDS variant of the LDS->global redirect.
;
; Triton (and other dynamic-shared-memory) kernels declare
; group_segment_fixed_size == 0 but still use LDS, sized at launch via
; sharedMemBytes.  The static redirect gate keys on a non-zero static size
; and so never fires for them.  This fixture pins the dynamic path:
;
;   - Detection is driven by the presence of redirectable DS storage ops,
;     NOT by a static size, and works under --enable-lds-redirect alone
;     (no --force-lds-redirect: there is no static size to compare to a cap).
;   - The lifted kernel gains BOTH the wg_lds_base ptr addrspace(1) arg AND
;     an i32 dyn_lds_g arg carrying the per-workgroup LDS byte size G, which
;     the interceptor fills from the launch's sharedMemBytes.
;   - The per-workgroup stride multiplies the linear workgroup id by the
;     RUNTIME dyn_lds_g (zext'd to i64), not a compile-time constant.
;   - DS storage redirects to ptr addrspace(1), and no amdgpu-lds-size
;     attribute is emitted.

; New wg_lds_base AND dyn_lds_g parameters.
; CHECK-LABEL: define amdgpu_kernel void @lds_redirect_dynamic_kernel(
; CHECK:       ptr addrspace(1) %wg_lds_base
; CHECK:       i32 %dyn_lds_g

; Stride uses the runtime G (zext of the dyn_lds_g kernarg), not a constant.
; CHECK: %lds_linear_wg_id =
; CHECK: %dyn_lds_g_64 = zext i32 %dyn_lds_g to i64
; CHECK: %lds_wg_byte_off = mul i64 %lds_linear_wg_id, %dyn_lds_g_64
; CHECK: %wg_lds_ptr = getelementptr inbounds i8

; DS store redirected to addrspace(1).
; CHECK: store i32 {{.*}}, ptr addrspace(1) %lds_global_ptr

; Barrier bracketing preserved.
; CHECK: fence syncscope("workgroup-one-as") release
; CHECK: call void @llvm.amdgcn.s.barrier()
; CHECK: fence syncscope("workgroup-one-as") acquire

; DS load redirected to addrspace(1).
; CHECK: load i32, ptr addrspace(1)

; No LDS attribute: the redirected kernel declares zero hardware LDS.
; CHECK-NOT: "amdgpu-lds-size"

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	lds_redirect_dynamic_kernel
	.p2align	8
	.type	lds_redirect_dynamic_kernel,@function
lds_redirect_dynamic_kernel:
	v_dual_add_nc_u32 v1, 1, v0 :: v_dual_lshlrev_b32 v2, 2, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	s_delay_alu instid0(VALU_DEP_1)
	v_and_b32_e32 v1, 31, v1
	ds_store_b32 v2, v0
	s_wait_dscnt 0x0
	s_barrier_signal -1
	s_barrier_wait -1
	v_lshlrev_b32_e32 v1, 2, v1
	ds_load_b32 v1, v1
	s_wait_dscnt 0x0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_redirect_dynamic_kernel
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
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
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           lds_redirect_dynamic_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         lds_redirect_dynamic_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
