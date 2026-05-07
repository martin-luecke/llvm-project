//===- opcode_map_test.cpp - opcode_map unit tests ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap/mc_state.h"
#include "hotswap/opcode_map.h"

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
