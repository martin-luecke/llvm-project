//===- opcode_map_test.cpp - opcode_map unit tests ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/mc_state.h"
#include "hotswap/opcode_map.h"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <mutex>

namespace {

void ensureAMDGPURegistered() {
  static std::once_flag Flag;
  std::call_once(Flag, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  });
}

} // namespace

// Empty map: every opcode should resolve to `CanonicalOp::Unknown` until
// handler patches start adding entries.
TEST(OpcodeMap, UnknownLookupBeforeBuild) {
  COMGR::hotswap::OpcodeMap Map;
  EXPECT_EQ(Map.lookup(0), COMGR::hotswap::CanonicalOp::Unknown);
  EXPECT_EQ(Map.lookup(12345), COMGR::hotswap::CanonicalOp::Unknown);
}

TEST(OpcodeMap, BuildOnGfx942IsBenign) {
  ensureAMDGPURegistered();
  COMGR::hotswap::MCState State;
  ASSERT_TRUE(COMGR::hotswap::initMCState(State, "gfx942"));

  COMGR::hotswap::OpcodeMap Map;
  Map.build(*State.InstrInfo);

  // No handler patches have landed yet, so every MC opcode resolves to
  // `Unknown` — the raiser bails on the first decoded instruction.
  EXPECT_EQ(Map.lookup(0), COMGR::hotswap::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250AddMinRealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState State;
  ASSERT_TRUE(COMGR::hotswap::initMCState(State, "gfx1250"));

  COMGR::hotswap::OpcodeMap Map;
  Map.build(*State.InstrInfo);

  EXPECT_EQ(Map.lookup(llvm::AMDGPU::V_ADD_MIN_U32_e64_gfx1250),
            COMGR::hotswap::CanonicalOp::V_ADD_MIN_U32);
}

TEST(OpcodeMap, Gfx1250Min3RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState State;
  ASSERT_TRUE(COMGR::hotswap::initMCState(State, "gfx1250"));

  COMGR::hotswap::OpcodeMap Map;
  Map.build(*State.InstrInfo);

  EXPECT_EQ(Map.lookup(llvm::AMDGPU::V_MIN3_U32_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MIN3_U32);
}

TEST(OpcodeMap, Gfx1250Dot4I32IU8RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState State;
  ASSERT_TRUE(COMGR::hotswap::initMCState(State, "gfx1250"));

  COMGR::hotswap::OpcodeMap Map;
  Map.build(*State.InstrInfo);

  EXPECT_EQ(Map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8),
            COMGR::hotswap::CanonicalOp::V_DOT4_I32_IU8);
  EXPECT_EQ(Map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8_gfx12),
            COMGR::hotswap::CanonicalOp::V_DOT4_I32_IU8);
}

TEST(OpcodeMap, Gfx1250PkFmaF16RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState State;
  ASSERT_TRUE(COMGR::hotswap::initMCState(State, "gfx1250"));

  COMGR::hotswap::OpcodeMap Map;
  Map.build(*State.InstrInfo);

  EXPECT_EQ(Map.lookup(llvm::AMDGPU::V_PK_FMA_F16_gfx12),
            COMGR::hotswap::CanonicalOp::V_PK_FMA_F16);
}

TEST(OpcodeMap, Gfx1250MadI32I24RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  COMGR::hotswap::MCState State;
  ASSERT_TRUE(COMGR::hotswap::initMCState(State, "gfx1250"));

  COMGR::hotswap::OpcodeMap Map;
  Map.build(*State.InstrInfo);

  EXPECT_EQ(Map.lookup(llvm::AMDGPU::V_MAD_I32_I24_e64_gfx12),
            COMGR::hotswap::CanonicalOp::V_MAD_I32_I24);
}
