; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco
; RUN: env HSA_HOTSWAP_STRICT=1 raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:   --emit-ir=kernarg_two_mode_implicitarg_strict \
; RUN:   | %FileCheck %s --check-prefix=STRICT

; Two-mode kernarg ABI (Tensile "UserArgs" shape). A runtime selector chooses
; between reading arguments inline from the entry kernarg segment (the pointer
; is rebased by a split 32-bit carry chain: s_add_co_u32 + s_add_co_ci_u32) and
; loading an indirect user-args buffer pointer (s_load_b64 into the pair). The
; two paths merge, so the kernarg pointer's provenance at the follow-up load is
; the recoverable EntryOrNonEntry state rather than hard-Unknown.
;
; The follow-up load's offset lands past the declared hidden-arg block, so it is
; not a modelled hidden field: it is an ordinary explicit/user-arg read that is
; correct on BOTH arms. Under strict mode the raiser must lower it as an
; ordinary memory load (no refusal, no implicitarg synthesis) rather than the
; hard-Unknown refusal it produced before the two-mode provenance was modelled.
; STRICT-LABEL: define amdgpu_kernel void @kernarg_two_mode_implicitarg_strict(
; STRICT-NOT: call ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
; STRICT: inttoptr i64
; STRICT: load i32, ptr addrspace(1)

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 5
	.text
	.globl	kernarg_two_mode_implicitarg_strict
	.p2align	8
	.type	kernarg_two_mode_implicitarg_strict,@function
kernarg_two_mode_implicitarg_strict:
	s_load_b32 s20, s[0:1], 0x0
	s_wait_kmcnt 0x0
	s_lshr_b32 s21, s20, 30
	s_cmp_eq_u32 s21, 0
	s_cbranch_scc0 .Lhbm
	s_add_co_u32 s0, s0, 16
	s_add_co_ci_u32 s1, s1, 0
	s_branch .Lmerge
.Lhbm:
	s_load_b64 s[0:1], s[0:1], 0x10
	s_wait_kmcnt 0x0
.Lmerge:
	s_load_b64 s[8:9], s[0:1], 0x90
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v0, s8
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel kernarg_two_mode_implicitarg_strict
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 1
		.amdhsa_next_free_sgpr 22
		.amdhsa_float_denorm_mode_32 3
	.end_amdhsa_kernel
	.text
	.p2alignl 7, 3214868480
	.fill 96, 4, 3214868480
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .offset:         0
        .size:           32
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 32
    .name:           kernarg_two_mode_implicitarg_strict
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .symbol:         kernarg_two_mode_implicitarg_strict.kd
    .vgpr_count:     1
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
