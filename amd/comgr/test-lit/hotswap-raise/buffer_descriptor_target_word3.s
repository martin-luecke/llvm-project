; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco --target-isa=gfx1151 \
; RUN:     --emit-ir=buffer_atomic_swap_b32_kernel 2>/dev/null \
; RUN:   | %FileCheck %s

; MUBUF buffer atomic on an RDNA target.
;
; The rebuilt V# must carry the target's raw-buffer format word3: both
; descriptor forms store that dword verbatim (make.buffer.rsrc operand 3 is not
; fixed up by SIISelLowering::lowerPointerAsRsrcIntrin). gfx10+ needs
; OOB_SELECT=3 for raw byte-extent bounds -- with the gfx9 encoding's
; OOB_SELECT=0 a stride-0 descriptor reports zero records, so every access is
; out of bounds and the atomic is dropped.
;
;   gfx11  0x31016000 == 822173696  UFMT_32_FLOAT | RESOURCE_LEVEL | OOB_SELECT=3
;
; The gfx942 counterpart of this kernel is refused before translation
; (non-commutative atomic under wave32->wave64 replica race), so the CDNA
; word3 is covered by buffer_store_short_srd_words.s instead.
;
; NUM_RECORDS comes from the shared gfx1250 source decode: dword2 alone is the
; byte extent >> 7, so it must be recombined with dword1[31:25] and dword3[5:0].

; CHECK-LABEL: define amdgpu_kernel void @buffer_atomic_swap_b32_kernel(
; CHECK: and i32 %{{.+}}, 127
; CHECK: shl i64 %{{.+}}, 7
; CHECK: shl i64 %{{.+}}, 39
; CHECK: %[[FULL:mubuf_raw_num_records_full[0-9]*]] = or i64
; CHECK: %[[NR:.+]] = select i1 %{{.+}}, i64 2147483646, i64 %[[FULL]]
; CHECK: %[[W2:.+]] = trunc i64 %[[NR]] to i32
; CHECK-NOT: insertelement <4 x i32> {{.*}}, i32 159744, i64 3
; CHECK: insertelement <4 x i32> %{{.+}}, i32 822173696, i64 3
; CHECK: call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %{{.+}}, i16 0, i64 %{{.+}}, i32 822173696)
; CHECK: call i32 @llvm.amdgcn.raw.buffer.atomic.swap

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	buffer_atomic_swap_b32_kernel
	.p2align	8
	.type	buffer_atomic_swap_b32_kernel,@function
buffer_atomic_swap_b32_kernel:          ; @buffer_atomic_swap_b32_kernel
; %bb.0:
	s_load_b96 s[4:6], s[0:1], 0x0
	v_lshlrev_b32_e32 v1, 2, v0
	s_mov_b32 s3, 0x27000
	s_mov_b32 s2, -1
	s_wait_kmcnt 0x0
	v_mov_b32_e32 v2, s6
	s_mov_b32 s0, s4
	s_mov_b32 s1, s5
	buffer_atomic_swap_b32 v2, v1, s[0:3], null offen th:TH_ATOMIC_RETURN scope:SCOPE_DEV
	s_wait_loadcnt 0
	
	global_store_b32 v0, v2, s[4:5] scale_offset
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel buffer_atomic_swap_b32_kernel
		.amdhsa_kernarg_size 12
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 3
		.amdhsa_next_free_sgpr 7
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
      - .offset:         8
        .size:           4
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 12
    .max_flat_workgroup_size: 1024
    .name:           buffer_atomic_swap_b32_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     7
    .symbol:         buffer_atomic_swap_b32_kernel.kd
    .vgpr_count:     3
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
