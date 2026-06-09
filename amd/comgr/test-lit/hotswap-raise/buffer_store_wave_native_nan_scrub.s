; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=buffer_store_wave_native_nan_scrub_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=WN
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --disable-wave-native \
; RUN:   --emit-ir=buffer_store_wave_native_nan_scrub_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=MR
;
; Regression guard for the WaveNative NaN-scrub on MUBUF scalar store data.
; Under WaveNative, source NaN-to-0 conversions before buffer_store may be
; dropped by the CFG walker; the lifter inserts a scrub unconditionally.
; Under ModuloReplication the source EXEC mask gates the store directly, so
; no scrub is inserted and the store sits inside an spe_skip diamond.

; WN-LABEL: define amdgpu_kernel void @buffer_store_wave_native_nan_scrub_kernel(
; WN: call i1 @llvm.amdgcn.init.whole.wave()
; Scalar path: bitcast i32 -> float, fcmp uno, select NaN -> 0.
; WN: %[[F:.+]] = bitcast i32 %{{.+}} to float
; WN: %[[UNO:.+]] = fcmp uno float %[[F]], %[[F]]
; WN: %[[SCRUB:.+]] = select i1 %[[UNO]], i32 0, i32 %{{.+}}
; WN: call void @llvm.amdgcn.raw.buffer.store.i32(i32 %[[SCRUB]],

; MR-LABEL: define amdgpu_kernel void @buffer_store_wave_native_nan_scrub_kernel(
; MR-NOT: fcmp uno
; MR: br i1 %{{[^,]+}}, label %spe_do{{[0-9]*}}, label %spe_skip{{[0-9]*}}
; MR: spe_do{{[0-9]*}}:
; MR: call void @llvm.amdgcn.raw.buffer.store.i32(

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_store_wave_native_nan_scrub_kernel
	.p2align	8
	.type	buffer_store_wave_native_nan_scrub_kernel,@function
buffer_store_wave_native_nan_scrub_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v1, 0
	s_mov_b32 s2, 16
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	buffer_store_dword v0, v1, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_wave_native_nan_scrub_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
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
    .max_flat_workgroup_size: 64
    .name:           buffer_store_wave_native_nan_scrub_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_wave_native_nan_scrub_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata

; Second RUN pair: vector path (buffer_store_dwordx4). Tests the ConstantVector
; getSplat branch in handleMUBUF -- the scalar RUN above only covers dwords==1.
; RUN: raise_cli %t.hsaco --target-isa=gfx942 --enable-wave-native \
; RUN:     --emit-ir=buffer_store_wave_native_nan_scrub_vec_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=VEC

; VEC-LABEL: define amdgpu_kernel void @buffer_store_wave_native_nan_scrub_vec_kernel(
; VEC: call i1 @llvm.amdgcn.init.whole.wave()
; Vector path: bitcast <4 x i32> -> <4 x float>, fcmp uno, select vector NaN -> zeroinitializer.
; VEC: %[[F:.+]] = bitcast <4 x i32> %{{.+}} to <4 x float>
; VEC: %[[UNO:.+]] = fcmp uno <4 x float> %[[F]], %[[F]]
; VEC: %[[SCRUB:.+]] = select <4 x i1> %[[UNO]], <4 x i32> zeroinitializer, <4 x i32> %{{.+}}
; VEC: call void @llvm.amdgcn.raw.buffer.store.v4i32(<4 x i32> %[[SCRUB]],

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_store_wave_native_nan_scrub_vec_kernel
	.p2align	8
	.type	buffer_store_wave_native_nan_scrub_vec_kernel,@function
buffer_store_wave_native_nan_scrub_vec_kernel:
	s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, 25, 1), 1
	s_load_b64 s[0:1], s[0:1], 0x0
	v_mov_b32_e32 v4, 0
	s_mov_b32 s2, 64
	s_mov_b32 s3, 0x27000
	s_wait_kmcnt 0x0
	buffer_store_dwordx4 v[0:3], v4, s[0:3], null offen
	s_wait_storecnt 0
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_store_wave_native_nan_scrub_vec_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 5
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
    .max_flat_workgroup_size: 64
    .name:           buffer_store_wave_native_nan_scrub_vec_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     4
    .symbol:         buffer_store_wave_native_nan_scrub_vec_kernel.kd
    .vgpr_count:     5
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
