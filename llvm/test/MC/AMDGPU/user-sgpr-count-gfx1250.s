// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -filetype=asm %s 2>&1 | FileCheck %s

.amdhsa_code_object_version 6

// CHECK:.amdhsa_user_sgpr_count 8
        .amdhsa_kernel user_sgpr_0
        .amdhsa_next_free_vgpr 0
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 8
.end_amdhsa_kernel

// CHECK:.amdhsa_user_sgpr_count 31
.amdhsa_kernel user_sgpr_1
        .amdhsa_next_free_vgpr 1
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 31
.end_amdhsa_kernel

// CHECK:.amdhsa_user_sgpr_count 32
.amdhsa_kernel user_sgpr_2
        .amdhsa_next_free_vgpr 1
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 32
        .end_amdhsa_kernel

// An explicit count below the count implied by the enabled user SGPRs is
// clamped up to the implied value, and the emitted descriptor carries the
// implied count. Here dispatch_ptr + queue_ptr + kernarg_segment_ptr +
// dispatch_id imply 8, so the explicit 2 is replaced by 8. Guards
// ParseDirectiveAMDHSAKernel in AMDGPUAsmParser.cpp.
// CHECK: warning: amdgpu_user_sgpr_count smaller than implied by enabled user SGPRs; using the implied count
// CHECK:.amdhsa_user_sgpr_count 8
.amdhsa_kernel user_sgpr_count_clamped_to_implied
        .amdhsa_next_free_vgpr 1
        .amdhsa_next_free_sgpr 0

        .amdhsa_user_sgpr_count 2
        .amdhsa_user_sgpr_dispatch_ptr 1
        .amdhsa_user_sgpr_queue_ptr 1
        .amdhsa_user_sgpr_kernarg_segment_ptr 1
        .amdhsa_user_sgpr_dispatch_id 1
.end_amdhsa_kernel
