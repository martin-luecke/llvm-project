; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:      --enable-lds-redirect --force-lds-redirect \
; RUN:      --emit-ir=lds_redirect_shape_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; IR-shape fixture for the LDS->global redirect path
; (HSA_HOTSWAP_LDS_TO_GLOBAL / HSA_HOTSWAP_LDS_TO_GLOBAL_FORCE).
;
; FORCE=1 applies the redirect even though the source LDS (4 KiB) is
; below gfx1151's 64 KiB cap, letting this fixture run independently
; of any real gfx1250 kernel that overflows the cap.
;
; Seven structural invariants are pinned:
;
;   1. The lifted kernel gains a new "wg_lds_base" ptr addrspace(1) arg.
;   2. A per-workgroup byte offset is computed from dispatch dims +
;      workgroup IDs (linear_wg_id * G).
;   3. Storage DS stores redirect to ptr addrspace(1) — the GEP result —
;      NOT LDS addrspace(3).
;   4. Every barrier is bracketed with release/acquire fences in
;      syncscope("workgroup-one-as") (-> s_wait_storecnt 0 /
;      buffer_gl0_inv on gfx1151 to maintain L0 coherence).
;   5. The s_barrier rendezvous intrinsic itself is still present.
;   6. Storage DS loads redirect to ptr addrspace(1), NOT addrspace(3).
;   7. The "amdgpu-lds-size" attribute is ABSENT (redirected kernels
;      declare zero LDS to the hardware).

; 1. New wg_lds_base parameter in addrspace(1).
; CHECK-LABEL: define amdgpu_kernel void @lds_redirect_shape_kernel(
; CHECK:       ptr addrspace(1) %wg_lds_base

; 2. linear_wg_id, per-wg byte offset, and the per-WG pointer GEP.
; CHECK: %lds_linear_wg_id =
; CHECK: %lds_wg_byte_off = mul i64
; CHECK: %wg_lds_ptr = getelementptr inbounds i8

; 3. DS store redirected to addrspace(1).
; CHECK: store i32 {{.*}}, ptr addrspace(1) %lds_global_ptr

; 4a. Release fence before the barrier.
; CHECK: fence syncscope("workgroup-one-as") release

; 5. The actual barrier rendezvous.
; CHECK: call void @llvm.amdgcn.s.barrier()

; 4b. Acquire fence after the barrier.
; CHECK: fence syncscope("workgroup-one-as") acquire

; 6. DS load redirected to addrspace(1).
; CHECK: load i32, ptr addrspace(1)

; 7. No LDS attribute: redirect zeroes the group_segment_fixed_size.
; CHECK-NOT: "amdgpu-lds-size"

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	lds_redirect_shape_kernel
	.p2align	8
	.type	lds_redirect_shape_kernel,@function
lds_redirect_shape_kernel:
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
	.amdhsa_kernel lds_redirect_shape_kernel
		.amdhsa_group_segment_fixed_size 4096
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
    .group_segment_fixed_size: 4096
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 32
    .name:           lds_redirect_shape_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         lds_redirect_shape_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
