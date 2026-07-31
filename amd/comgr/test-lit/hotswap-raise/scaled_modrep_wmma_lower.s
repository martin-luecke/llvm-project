; A WMMA kernel with a workitem.id.y divergent early-exit (the attention
; _fwd_kernel class) auto-upgrades to a scaled dispatch and lowers to MFMA fed by
; active replica lanes, with no whole-wave scaffolding. Guards the scaled-dispatch
; matrix path: wwmMatrixCollective in wmma-lowering.cpp and the y/z-refusal
; auto-upgrade in raiser.cpp. max_flat_workgroup_size=256 so the scaled block
; (512) fits the target max. See hotswap/docs/modrep-predicate-chain.md sec. 10.

; The auto-upgrade route and --force reach the same projection and IR. The
; --implicit-check-not asserts the matrix collective is NOT wrapped in strict.wwm
; (a scaled dispatch keeps all lanes active, so the wrap is unneeded -- and on a
; large tile it overflows SIPreAllocateWWMRegs and crashes codegen).
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=scaled_modrep_wmma_lower_kernel 2>&1 \
; RUN:   | %FileCheck %s --implicit-check-not=strict.wwm
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_wmma_lower_kernel 2>&1 \
; RUN:   | %FileCheck %s --implicit-check-not=strict.wwm
; CHECK: selected ScaledModuloReplicationProjection
; CHECK: define amdgpu_kernel void @scaled_modrep_wmma_lower_kernel(
; CHECK-NOT: init_whole_wave

; Codegen (--write-hsaco) is the stage the strict.wwm wrap regressed; require it
; to succeed.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --write-hsaco=%t.co --kernel=scaled_modrep_wmma_lower_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=HSACO
; HSACO: wrote {{.+}} byte HSACO for kernel 'scaled_modrep_wmma_lower_kernel'

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	scaled_modrep_wmma_lower_kernel
	.p2align	8
	.type	scaled_modrep_wmma_lower_kernel,@function
scaled_modrep_wmma_lower_kernel:
	s_load_b64 s[2:3], s[0:1], 0x0
	v_bfe_u32 v30, v0, 10, 10
	s_wait_kmcnt 0x0
	v_cmp_lt_u32_e64 s4, v30, 16
	v_cndmask_b32_e64 v24, -1, v30, s4
; the WMMA lowers to the MFMA redistribute collective:
; CHECK: call {{.+}}@llvm.amdgcn.mfma
	v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
	global_store_b128 v24, v[16:19], s[2:3]
	s_endpgm
; the scaled dispatch is advertised on the raised kernel (module-end attrs):
; CHECK-DAG: "amdgpu-flat-work-group-size"="512,512"
; CHECK-DAG: "hotswap-scaled-dispatch"="x2"
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel scaled_modrep_wmma_lower_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_vgpr_workitem_id 1
		.amdhsa_next_free_vgpr 32
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
    .max_flat_workgroup_size:  256
    .name:                     scaled_modrep_wmma_lower_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         scaled_modrep_wmma_lower_kernel.kd
    .vgpr_count:     32
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
