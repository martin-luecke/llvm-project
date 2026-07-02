; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_mov_b16_lo_lo_kernel 2>/dev/null | %FileCheck %s --check-prefix=LOLO
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_mov_b16_hi_lo_kernel 2>/dev/null | %FileCheck %s --check-prefix=HILO
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_mov_b16_lo_hi_kernel 2>/dev/null | %FileCheck %s --check-prefix=LOHI
; RUN: %raise_cli %t.hsaco --target-isa=gfx942 --emit-ir=v_mov_b16_hi_hi_kernel 2>/dev/null | %FileCheck %s --check-prefix=HIHI

; LOLO-LABEL: define amdgpu_kernel void @v_mov_b16_lo_lo_kernel(
; LOLO: trunc i32 {{.*}} to i16
; LOLO: zext i16 {{.*}} to i32
; LOLO: and i32 {{.*}}, -65536
; LOLO: %v_mov_b16_merge{{.*}} = or i32
; LOLO-NOT: lshr i32 {{.*}}, 16
; LOLO-NOT: shl i32 {{.*}}, 16
; HILO-LABEL: define amdgpu_kernel void @v_mov_b16_hi_lo_kernel(
; HILO: lshr i32 {{.*}}, 16
; HILO: trunc i32 {{.*}} to i16
; HILO: zext i16 {{.*}} to i32
; HILO: and i32 {{.*}}, -65536
; HILO: %v_mov_b16_merge{{.*}} = or i32
; HILO-NOT: shl i32 {{.*}}, 16
; LOHI-LABEL: define amdgpu_kernel void @v_mov_b16_lo_hi_kernel(
; LOHI: trunc i32 {{.*}} to i16
; LOHI: zext i16 {{.*}} to i32
; LOHI: and i32 {{.*}}, 65535
; LOHI: shl i32 {{.*}}, 16
; LOHI: %v_mov_b16_merge{{.*}} = or i32
; LOHI-NOT: lshr i32 {{.*}}, 16
; HIHI-LABEL: define amdgpu_kernel void @v_mov_b16_hi_hi_kernel(
; HIHI: lshr i32 {{.*}}, 16
; HIHI: trunc i32 {{.*}} to i16
; HIHI: zext i16 {{.*}} to i32
; HIHI: and i32 {{.*}}, 65535
; HIHI: shl i32 {{.*}}, 16
; HIHI: %v_mov_b16_merge{{.*}} = or i32

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	v_mov_b16_lo_lo_kernel
	.p2align	8
	.type	v_mov_b16_lo_lo_kernel,@function
v_mov_b16_lo_lo_kernel:
	v_mov_b16 v0.l, v1.l
	s_endpgm

	.globl	v_mov_b16_hi_lo_kernel
	.p2align	8
	.type	v_mov_b16_hi_lo_kernel,@function
v_mov_b16_hi_lo_kernel:
	v_mov_b16 v0.l, v1.h
	s_endpgm

	.globl	v_mov_b16_lo_hi_kernel
	.p2align	8
	.type	v_mov_b16_lo_hi_kernel,@function
v_mov_b16_lo_hi_kernel:
	v_mov_b16 v0.h, v1.l
	s_endpgm

	.globl	v_mov_b16_hi_hi_kernel
	.p2align	8
	.type	v_mov_b16_hi_hi_kernel,@function
v_mov_b16_hi_hi_kernel:
	v_mov_b16 v0.h, v1.h
	s_endpgm

	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel v_mov_b16_lo_lo_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_mov_b16_hi_lo_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_mov_b16_lo_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_inst_pref_size 1
	.end_amdhsa_kernel
	.p2align	6, 0x0
	.amdhsa_kernel v_mov_b16_hi_hi_kernel
		.amdhsa_kernarg_size 0
		.amdhsa_user_sgpr_count 0
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 8
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
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_mov_b16_lo_lo_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mov_b16_lo_lo_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_mov_b16_hi_lo_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mov_b16_hi_lo_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_mov_b16_lo_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mov_b16_lo_hi_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
  - .args: []
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 0
    .max_flat_workgroup_size: 1024
    .name:           v_mov_b16_hi_hi_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     8
    .symbol:         v_mov_b16_hi_hi_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
