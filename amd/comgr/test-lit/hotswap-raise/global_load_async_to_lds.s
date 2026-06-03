; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx942 --emit-ir=global_load_async_to_lds_kernel 2>/dev/null \
; RUN:   | %FileCheck %s --check-prefix=IR
;
; The runtime LDS-dest gate reads HW_REG_LDS_ALLOC.LDS_SIZE, whose layout is
; only verified for gfx9. Other targets refuse rather than gate on an
; unverified encoding. Pin that refusal on a gfx11 target.
;
; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && %not raise_cli %t.hsaco \
; RUN:     --target-isa=gfx1100 --emit-ir=global_load_async_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=REFUSE
;
; REFUSE: failed to raise: global_load_async_to_lds_b32 [FLAT]
; REFUSE-SAME: HW_REG_LDS_ALLOC.LDS_SIZE layout not verified for this target
;
; Lift fixture for FLAT `global_load_async_to_lds_b{8,32,64,128}` on the
; CROSS-TARGET arm (gfx1250 -> gfx942), where `hasTensorOps` is false. The
; cross-target arm has no async DMA, so the raiser emulates it with a
; synchronous per-lane `load` + `store`: bit-identical per-lane LDS state to the
; async DMA after `s_wait_asynccnt 0`, losing only pipelining overlap (a
; throughput, not correctness, regression).
;
; `scale_offset` (per-lane VGPR offset = tid * elemBytes) is materialised as
; `mul i64 %voff_zext, N` for N = 1/4/8/16. b8 has N=1, so the multiply is
; elided. Widths > b32 are lifted as `<n x i32>` so the backend picks the right
; dwordx{2,4} / ds_store opcode from the attached natural alignment:
;   b32: i32   b64: <2 x i32>   b128: <4 x i32>   b8: i8

; LDS-destination drop gate. gfx12 drops a lane whose LDS-dest offset is past
; the LDS allocation; Triton uses an INT_MAX sentinel to predicate masked
; GEMM-tile rows (intentionally-OOB global address). The emulation replicates
; the drop by reading the allocation at runtime: `s_getreg` simm16 17158 =
; hwreg(LDS_ALLOC, 12, 9), granules * 512 bytes, then `icmp ult` + `br`.
;
; ----- b32 ----- (first load in the HIP kernel)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 4
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %lds_alloc_granules{{[0-9]*}} = call i32 @llvm.amdgcn.s.getreg(i32 17158)
; IR: %lds_alloc_bytes{{[0-9]*}} = mul i32 %lds_alloc_granules{{[0-9]*}}, 512
; IR: %async_lds_inb{{[0-9]*}} = icmp ult i32 {{.*}}, %lds_alloc_bytes{{[0-9]*}}
; IR: br i1 %async_lds_inb{{[0-9]*}}
; IR: %async_gload{{[0-9]*}} = load i32, ptr addrspace(1) %{{[0-9]+}}, align 4
; IR: store i32 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 4

; ----- b64 ----- (second load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 8
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <2 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 8
; IR: store <2 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 8

; ----- b128 ----- (third load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %scaled_voff{{[0-9]*}} = mul i64 %voff_zext{{[0-9]*}}, 16
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %scaled_voff{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load <4 x i32>, ptr addrspace(1) %{{[0-9]+}}, align 16
; IR: store <4 x i32> %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 16

; ----- b8 ----- (fourth load)

; IR: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; IR: %voff_zext{{[0-9]*}} = zext i32 {{.*}} to i64
; IR: %saddr_vaddr{{[0-9]*}} = add i64 {{.*}}, %voff_zext{{[0-9]*}}
; IR: %{{[0-9]+}} = inttoptr i64 %saddr_vaddr{{[0-9]*}} to ptr addrspace(1)
; IR: %async_gload{{[0-9]*}} = load i8, ptr addrspace(1) %{{[0-9]+}}, align 1
; IR: store i8 %async_gload{{[0-9]*}}, ptr addrspace(3) %lds_ptr{{[0-9]*}}, align 1

; ----- Negative assertions -----
;
; The cross-target arm MUST NOT emit the gfx1250-only intrinsic (it would fail
; isel on gfx942).

; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b8
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b32
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b64
; IR-NOT: @llvm.amdgcn.global.load.async.to.lds.b128

; RUN: %llvm_mc -mcpu=gfx1250 %s -o %t.o && %ld_lld -shared %t.o -o %t.hsaco \
; RUN:   && raise_cli %t.hsaco \
; RUN:     --target-isa=gfx1250 --emit-ir=global_load_async_to_lds_kernel 2>&1 \
; RUN:   | %FileCheck %s --check-prefix=SAME
;
; Same-target (gfx1250 -> gfx1250) intrinsic-emit path, where `hasTensorOps` is
; true. The handler casts `vdst` to `ptr addrspace(3)` (`lds_ptr*`), decodes the
; global address, threads the FLAT `offset` and `cpol` immediates through, and
; wraps the call in `emitUnderExec`:
;
;   void llvm.amdgcn.global.load.async.to.lds.b{8,32,64,128}(
;       ptr addrspace(1) %gaddr, ptr addrspace(3) %laddr,
;       i32 immarg %offset, i32 immarg %cpol)
;
; Pins the per-width call shape and the LDS-pointer cast it consumes, catching
; an intrinsic rename or a non-`ptr addrspace(3)` LDS-base lowering.

; b32: per-lane LDS i32 base via inttoptr i32 → ptr addrspace(3),
; then the b32 async DMA call.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b32(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

; b64: same shape, b64 intrinsic.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b64(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

; b128: same shape, b128 intrinsic.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b128(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

; b8: same shape, b8 intrinsic.
; SAME: %lds_ptr{{[0-9]*}} = inttoptr i32 {{.*}} to ptr addrspace(3)
; SAME: call void @llvm.amdgcn.global.load.async.to.lds.b8(
; SAME-SAME: ptr addrspace(1)
; SAME-SAME: ptr addrspace(3) %lds_ptr
; SAME-SAME: i32 0
; SAME-SAME: i32 {{-?[0-9]+}}

	.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"
	.amdhsa_code_object_version 6
	.text
	.globl	global_load_async_to_lds_kernel
	.p2align	8
	.type	global_load_async_to_lds_kernel,@function
global_load_async_to_lds_kernel:        ; @global_load_async_to_lds_kernel
; %bb.0:
	s_load_b256 s[4:11], s[0:1], 0x0
	v_lshl_add_u32 v1, v0, 2, 0x600
	s_wait_kmcnt 0x0
	global_load_async_to_lds_b32 v1, v0, s[4:5] scale_offset
	v_lshl_add_u32 v1, v0, 3, 0x400
	global_load_async_to_lds_b64 v1, v0, s[6:7] scale_offset
	v_lshlrev_b32_e32 v1, 4, v0
	global_load_async_to_lds_b128 v1, v0, s[8:9] scale_offset
	v_add_nc_u32_e32 v1, 0x700, v0
	global_load_async_to_lds_b8 v1, v0, s[10:11]
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel global_load_async_to_lds_kernel
		.amdhsa_group_segment_fixed_size 1856
		.amdhsa_kernarg_size 32
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_wavefront_size32 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_next_free_vgpr 2
		.amdhsa_next_free_sgpr 12
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
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 1856
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .max_flat_workgroup_size: 1024
    .name:           global_load_async_to_lds_kernel
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .symbol:         global_load_async_to_lds_kernel.kd
    .vgpr_count:     2
    .wavefront_size: 32
amdhsa.target:   amdgcn-amd-amdhsa--gfx1250
amdhsa.version: [1, 2]
...

	.end_amdgpu_metadata
