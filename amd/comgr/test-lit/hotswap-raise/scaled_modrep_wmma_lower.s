; A matrix (WMMA) kernel that ALSO carries a workitem.id.y()-derived divergent
; early-exit (the gemma attention _fwd_kernel class: a matmul body under a
; sequence/row-tail predicate that WaveNative wave32->wave64 packing cannot
; represent -- the divergent partner wave leaks a stale base VGPR into a load).
; With no flag or env the raiser auto-upgrades the WaveNative y/z refusal to
; ScaledModuloReplicationProjection, and the WMMA lowers correctly under it: the
; WMMA->MFMA redistribute's numSourceWavesPerTarget()==1 path computes the exact
; source-wave result and replicates it to the upper lanes, and the scaled
; dispatch keeps those upper lanes active so the Wave64 MFMA collective is valid.
; No init_whole_wave, no partner wave. See
; hotswap/docs/modrep-predicate-chain.md sec. 10.
;
; max_flat_workgroup_size=256 so the scaled block (512) fits the gfx942 max.

; Default (auto-upgrade), and --force-scaled-modrep, reach the same projection:
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=scaled_modrep_wmma_lower_kernel 2>&1 \
; RUN:   | %FileCheck %s
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_wmma_lower_kernel 2>&1 \
; RUN:   | %FileCheck %s

; The matrix kernel is NOT refused; it auto-upgrades and lowers with MFMA.
; CHECK: selected ScaledModuloReplicationProjection
; CHECK: define amdgpu_kernel void @scaled_modrep_wmma_lower_kernel(
; the WMMA lowered to the MFMA redistribute (matrix under a scaled dispatch):
; CHECK: call {{.*}}@llvm.amdgcn.mfma
; no WaveNative whole-wave forcing and no partner wave:
; CHECK-NOT: init_whole_wave
; the dispatch is advertised scaled and the shim breadcrumb is emitted:
; CHECK-DAG: "amdgpu-flat-work-group-size"="512,512"
; CHECK-DAG: "hotswap-scaled-dispatch"="x2"

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
	v_wmma_f32_16x16x32_f16 v[16:23], v[0:7], v[8:15], v[16:23]
	global_store_b128 v24, v[16:19], s[2:3]
	s_endpgm
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
