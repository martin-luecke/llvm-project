; A matrix (WMMA) kernel with a workitem.id.y()-derived divergent early-exit
; whose source block is 640 threads: scaling it needs 1280 threads/block, past
; the gfx942 hardware max, so the scaled route is ineligible and the WaveNative
; y/z refusal stands. Matrix ops are NOT special-cased -- the block-size gate is
; the same for matrix and non-matrix kernels (a matrix kernel that FITS lowers,
; see scaled_modrep_wmma_lower.s). This proves a too-large matrix kernel refuses
; loudly rather than miscomputing. See
; hotswap/docs/modrep-predicate-chain.md sec. 10.4.

; Default (auto-upgrade): the size gate makes the scaled route ineligible, and a
; matrix kernel is ineligible for the ThreadLoop C5 retry (it needs all target
; lanes at once), so the principled WaveNative y/z refusal stands rather than
; scaling past the hardware max.
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && not raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=scaled_modrep_wmma_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=NOUPGRADE
; NOUPGRADE: workitem.id.y()/.z()-derived predicate under WaveNative
; NOUPGRADE-NOT: selected ScaledModuloReplicationProjection
; NOUPGRADE-NOT: define amdgpu_kernel void @scaled_modrep_wmma_refuse_kernel(

; --force-scaled-modrep: refuse with the size-gate diagnostic (not a matrix
; diagnostic -- matrix is allowed; only the scaled block size is the obstruction).
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && not raise_cli %t.hsaco --target-isa=gfx942 --force-scaled-modrep \
; RUN:     --emit-ir=scaled_modrep_wmma_refuse_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SIZE
; SIZE: ScaledModuloReplicationProjection needs to launch 1280 thread
; SIZE-SAME: target hardware limit is 1024
; SIZE-NOT: define amdgpu_kernel void @scaled_modrep_wmma_refuse_kernel(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.text
	.globl	scaled_modrep_wmma_refuse_kernel
	.p2align	8
	.type	scaled_modrep_wmma_refuse_kernel,@function
scaled_modrep_wmma_refuse_kernel:
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
	.amdhsa_kernel scaled_modrep_wmma_refuse_kernel
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
    .max_flat_workgroup_size:  640
    .name:                     scaled_modrep_wmma_refuse_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .symbol:         scaled_modrep_wmma_refuse_kernel.kd
    .vgpr_count:     32
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
