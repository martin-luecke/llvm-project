; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx942 \
; RUN:     --emit-ir=ttmp6_init_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; Regression-fence for ttmp6 raiser-entry initialisation on gfx12+.
;
; On gfx12+ the command processor stores per-wave dispatch metadata in
; ttmp6:
;   bits[3:0]   = wave_id_in_threadgroup
;   bits[15:12] = max_wave_id (num_waves_per_wg - 1)
;
; Triton-generated kernels read these fields via `s_bfe_u32` and
; `s_and_b32` to compute global wave indices for persistent-schedule
; dispatch. Without initialisation reads of ttmp6 return LLVM undef,
; which can poison downstream computations through LLVM's undef
; propagation semantics even when the value is on the dead side of
; a select.
;
; This test assembles a minimal kernel that reads both ttmp6 fields
; and verifies the raised IR contains the expected initialisation:
;   - workitem.id.x call (to derive wave_id)
;   - lshr by 5 (divide by source wave size to get wave_id)
;   - and with 0xF (mask to 4 bits for bits[3:0])
;   - or to combine with the shifted max_wave_id
;
; The kernel reads s0 = bfe(ttmp6, 12, 4) and s1 = ttmp6 & 15, then
; uses both in a scalar add whose result gates a store. If ttmp6 were
; undef the add result would be undef and the store address would be
; garbage.

; IR-LABEL: define amdgpu_kernel void @ttmp6_init_kernel(
;
; The ttmp6 initialisation derives wave_id from workitem.id.x:
; IR:       %ttmp6_tid = call i32 @llvm.amdgcn.workitem.id.x()
; IR:       %ttmp6_wave_id = lshr i32 %ttmp6_tid, 5
; IR:       %ttmp6_wave_id_lo4 = and i32 %ttmp6_wave_id, 15
;
; The max_wave_id field comes from the kernel metadata
; (max_flat_workgroup_size / wave_size - 1), shifted into bits[15:12]:
; IR:       %ttmp6_val = or i32 %ttmp6_wave_id_lo4,

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	ttmp6_init_kernel
	.p2align	8
	.type	ttmp6_init_kernel,@function
ttmp6_init_kernel:
	s_load_dwordx2 s[2:3], s[0:1], 0x0
	; Read num_waves-1 from ttmp6[15:12]
	s_bfe_u32 s0, ttmp6, 0x4000c
	; Read wave_id from ttmp6[3:0]
	s_and_b32 s1, ttmp6, 15
	; Combine: global_wave_id = wave_id + wg_id * num_waves
	s_add_co_i32 s0, s0, 1
	s_mul_i32 s0, ttmp9, s0
	s_add_co_i32 s0, s1, s0
	; Use the result as a store offset
	v_mov_b32_e32 v1, s0
	v_lshlrev_b32_e32 v0, 2, v0
	s_wait_kmcnt 0x0
	global_store_b32 v0, v1, s[2:3] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel ttmp6_init_kernel
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 4
	.end_amdhsa_kernel
	.text
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - { .address_space: global, .offset: 0, .size: 8, .value_kind: global_buffer }
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 128
    .name: ttmp6_init_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 4
    .symbol: ttmp6_init_kernel.kd
    .vgpr_count: 2
    .wavefront_size: 32
amdhsa.version: [1, 2]
...
	.end_amdgpu_metadata
