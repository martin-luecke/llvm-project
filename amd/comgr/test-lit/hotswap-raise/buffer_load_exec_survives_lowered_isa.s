; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --isa=gfx1250 --target-isa=gfx950 \
; RUN:     --write-hsaco=%t.out.hsaco --kernel=buffer_load_exec_survives_kernel \
; RUN:   && %llvm-objdump -d --mcpu=gfx950 %t.out.hsaco | %FileCheck %s

; LOWERED-ISA regression guard (Martin review: "test the FINAL LOWERED ISA
; preserves EXEC masking, not just the raised IR" -- a raised-IR FileCheck
; passes even if the backend re-flattens the guard).
;
; Under wave-native wave32 -> wave64 the Option 1 emitGuardedMemOp primitive
; lowers the per-lane load guard through llvm.amdgcn.if / llvm.amdgcn.end.cf
; (SI_IF / SI_END_CF pseudos, marked side-effecting). The AMDGPU backend must
; therefore keep the EXEC save / conditional-branch / restore around the
; buffer_load instead of if-converting it to an unconditional EXEC = -1 load.
; This fixture disassembles the FINAL gfx950 HSACO and asserts that control
; flow survives: s_and_saveexec + s_cbranch_execz bracket the buffer_load.
;
; GOTCHA (from the #277/#278 review eval): at O2 a load whose result is unused
; is DCE'd entirely (the guard + mem op vanish), making a naive check vacuous.
; This kernel therefore CONSUMES the loaded value by storing it back to a
; second buffer, so the load + its guard must survive to ISA.

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_load_exec_survives_kernel
	.p2align	8
	.type	buffer_load_exec_survives_kernel,@function
buffer_load_exec_survives_kernel:
	s_load_b64 s[0:1], s[0:1], 0x0
	v_lshlrev_b32_e32 v0, 4, v0
	s_or_b32 s1, s1, 0xfc000000
	s_mov_b32 s3, 0
	s_mov_b32 s2, 0xffffff
	s_wait_kmcnt 0x0
	buffer_load_b128 v[4:7], v0, s[0:3], null offen
	s_wait_loadcnt 0
	; consume the loaded value so it is not DCE'd: store it back
	buffer_store_b128 v[4:7], v0, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm

; The EXEC-masking control flow survives to final gfx950 ISA around the
; buffer LOAD (the whole point of the backend-respected emitGuardedMemOp
; lowering). The guard opens with s_and_saveexec_b64 and is closed by an
; EXEC-conditional branch -- s_cbranch_execz for the plain diamond, or the
; equivalent s_xor exec / s_cbranch_execnz hammock form the backend may pick
; when the guarded (loaded) value is consumed by a following op (as here: the
; load is stored back to defeat DCE). Either way the buffer_load executes only
; under the saved EXEC mask and is NOT if-converted to an unconditional
; exec=-1 access. The trailing buffer_store proves the load survived (was not
; DCE'd). NOTE (#277 scope): the STORE hardening lands in the stacked #278; on
; #277 alone the store may lower unguarded, so this fixture asserts only the
; LOAD guard survival.
; CHECK-LABEL: <buffer_load_exec_survives_kernel>:
; CHECK: s_and_saveexec_b64
; CHECK: buffer_load
; CHECK: s_cbranch_exec{{[nz]+}}
; CHECK: buffer_store

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_load_exec_survives_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 4
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
      - { .address_space:  global, .offset:         0, .size:           8, .value_kind:     global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           buffer_load_exec_survives_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_load_exec_survives_kernel.kd
    .vgpr_count:     8
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
