; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:      --enable-lds-redirect --force-lds-redirect \
; RUN:      --emit-ir=lds_redirect_xlan_kernel 2>/dev/null \
; RUN:   | %FileCheck %s
;
; Cross-lane exclusion fixture for the LDS->global redirect path.
;
; Property under test: when the LDS->global redirect is active, crossbar
; instructions (ds_swizzle_b32) must NOT be redirected to global memory.
; They use the LDS crossbar for lane shuffles but allocate no LDS storage,
; so they are orthogonal to the storage redirect and must continue to emit
; their original intrinsics unchanged.
;
; At the same time, storage DS instructions (ds_store_b32, ds_load_b32) in
; the same kernel MUST be redirected to addrspace(1).
;
; Two invariants are pinned:
;
;   1. ds_swizzle_b32 -> llvm.amdgcn.ds.swizzle (unchanged crossbar intrinsic).
;      It must NOT become a global load/store — it has no storage backing.
;
;   2. ds_store_b32 / ds_load_b32 -> store/load ptr addrspace(1)
;      (redirected from addrspace(3) to the wg_lds_base global allocation).
;
; The BITMASK_PERM offset 0x041F (XOR_MASK=1, SWAP-pairs pattern) is the
; same value as c2_ds_swizzle.s, keeping the swizzle sub-mode consistent
; with the existing crossbar test corpus.

; Checks are ordered to match IR text layout: the raiser emits SPE
; (exec-emulation) blocks per instruction, so ds_store_b32 (spe_do13)
; precedes the swizzle call (spe_skip14), which precedes ds_load_b32
; (spe_do31). Intrinsic declares appear at the end of the module.

; CHECK-LABEL: define amdgpu_kernel void @lds_redirect_xlan_kernel(

; 2a. DS store redirected to addrspace(1) (first storage op in IR text).
; CHECK: store i32 {{.*}}, ptr addrspace(1) %lds_global_ptr

; 1. Crossbar swizzle: lifted to the ds.swizzle intrinsic, not global memory.
;    It must NOT become a global load/store — it has no storage backing.
; CHECK: call i32 @llvm.amdgcn.ds.swizzle(i32 %{{[^,]+}}, i32 1055)

; 2b. DS load redirected to addrspace(1) (follows swizzle in IR text).
; CHECK: load i32, ptr addrspace(1)

; The swizzle intrinsic declaration must be present (required by LLVM).
; CHECK: declare {{.*}}i32 @llvm.amdgcn.ds.swizzle(i32, i32 immarg{{.*}})

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	lds_redirect_xlan_kernel
	.p2align	8
	.type	lds_redirect_xlan_kernel,@function
lds_redirect_xlan_kernel:
	; v0 = lane_id (from hardware)
	; v1 = byte_offset = lane_id * 4
	v_lshlrev_b32_e32 v1, 2, v0
	s_load_b64 s[0:1], s[0:1], 0x0
	; Storage: write lane_id into LDS[byte_offset].
	; Redirect active -> becomes global store via wg_lds_base.
	ds_store_b32 v1, v0
	s_wait_dscnt 0x0
	; Crossbar: shuffle v0 across lanes using BITMASK_PERM XOR_MASK=1.
	; No LDS storage allocated -- wg_lds_base is irrelevant here.
	; The raise must emit llvm.amdgcn.ds.swizzle, not a global load.
	ds_swizzle_b32 v2, v0 offset:0x041f
	s_wait_dscnt 0x0
	; Storage: read LDS[byte_offset] back.
	; Redirect active -> becomes global load via wg_lds_base.
	ds_load_b32 v3, v1
	s_wait_dscnt 0x0
	v_add_u32_e32 v2, v2, v3
	s_wait_kmcnt 0x0
	global_store_b32 v0, v2, s[0:1] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel lds_redirect_xlan_kernel
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
    .name:           lds_redirect_xlan_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     2
    .symbol:         lds_redirect_xlan_kernel.kd
    .vgpr_count:     4
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
