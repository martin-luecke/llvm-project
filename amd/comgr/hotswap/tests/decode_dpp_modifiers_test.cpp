#include "../decode.hpp"
#include "../mc_state.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIDefines.h"

#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <mutex>
#include <string>

namespace {

std::once_flag RegisterOnce;

void ensureAMDGPURegistered() {
  std::call_once(RegisterOnce, []() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUDisassembler();
  });
}

llvm::MCInst makeTemplateInst(const llvm::MCInstrInfo &II, unsigned opc) {
  llvm::MCInst inst;
  inst.setOpcode(opc);
  const llvm::MCInstrDesc &desc = II.get(opc);
  for (unsigned i = 0; i < desc.getNumOperands(); ++i) {
    const llvm::MCOperandInfo &oi = desc.operands()[i];
    if (oi.OperandType == llvm::MCOI::OPERAND_IMMEDIATE || oi.isGenericImm()) {
      inst.addOperand(llvm::MCOperand::createImm(0));
    } else {
      inst.addOperand(llvm::MCOperand::createReg(llvm::AMDGPU::VGPR0));
    }
  }
  return inst;
}

void setNamedImm(llvm::MCInst &inst, unsigned opc, llvm::AMDGPU::OpName name,
                 int64_t v) {
  int idx = llvm::AMDGPU::getNamedOperandIdx(opc, name);
  ASSERT_GE(idx, 0);
  ASSERT_LT(idx, static_cast<int>(inst.getNumOperands()));
  ASSERT_TRUE(inst.getOperand(static_cast<unsigned>(idx)).isImm());
  inst.getOperand(static_cast<unsigned>(idx)).setImm(v);
}

transpiler::DecodedInst
makeDi(llvm::MCInst inst, uint64_t tsFlags) {
  transpiler::DecodedInst di;
  di.inst = std::move(inst);
  di.rawMnemonic = "unit_test";
  di.tsFlags = tsFlags;
  return di;
}

} // namespace

TEST(DecodeDppModifiers, NoDppTsFlagIsNoOp) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  const unsigned opc = llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp_gfx12;
  llvm::MCInst inst = makeTemplateInst(*state.instrInfo, opc);
  transpiler::DecodedInst di =
      makeDi(std::move(inst), /*tsFlags=*/0);
  transpiler::decodeDppModifiers(di);
  EXPECT_FALSE(di.hasDpp);
}

TEST(DecodeDppModifiers, Dpp16_DecodesFiZero) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  const unsigned opc = llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp_gfx12;
  llvm::MCInst inst = makeTemplateInst(*state.instrInfo, opc);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::dpp_ctrl, 0x100);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::row_mask, 9);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::bank_mask, 10);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::bound_ctrl, 0);
  int fiIdx = llvm::AMDGPU::getNamedOperandIdx(opc, llvm::AMDGPU::OpName::fi);
  ASSERT_GE(fiIdx, 0);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::fi, 0);

  const uint64_t ts = state.instrInfo->get(opc).TSFlags;
  ASSERT_NE(ts & llvm::SIInstrFlags::DPP, 0u);

  transpiler::DecodedInst di = makeDi(std::move(inst), ts);
  transpiler::decodeDppModifiers(di);
  EXPECT_TRUE(di.hasDpp);
  EXPECT_FALSE(di.dppFi);
  EXPECT_EQ(di.dppCtrl, static_cast<uint16_t>(0x100));
  EXPECT_EQ(di.dppRowMask, static_cast<uint8_t>(9 & 0xF));
  EXPECT_EQ(di.dppBankMask, static_cast<uint8_t>(10 & 0xF));
  EXPECT_FALSE(di.dppBoundCtrl);
}

TEST(DecodeDppModifiers, Dpp16_DecodesFiOne) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  const unsigned opc = llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp_gfx12;
  llvm::MCInst inst = makeTemplateInst(*state.instrInfo, opc);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::dpp_ctrl, 0);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::row_mask, 0xF);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::bank_mask, 0xF);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::bound_ctrl, 0);
  setNamedImm(inst, opc, llvm::AMDGPU::OpName::fi, 1);

  const uint64_t ts = state.instrInfo->get(opc).TSFlags;
  transpiler::DecodedInst di = makeDi(std::move(inst), ts);
  transpiler::decodeDppModifiers(di);
  EXPECT_TRUE(di.hasDpp);
  EXPECT_TRUE(di.dppFi);
}

TEST(DecodeDppModifiers, Dpp8_LeavesHasDppFalse) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  const unsigned opc = llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp8_gfx12;
  EXPECT_GE(llvm::AMDGPU::getNamedOperandIdx(opc, llvm::AMDGPU::OpName::dpp8),
            0);
  llvm::MCInst inst = makeTemplateInst(*state.instrInfo, opc);

  const uint64_t ts = state.instrInfo->get(opc).TSFlags;
  ASSERT_NE(ts & llvm::SIInstrFlags::DPP, 0u);

  transpiler::DecodedInst di = makeDi(std::move(inst), ts);
  transpiler::decodeDppModifiers(di);
  EXPECT_FALSE(di.hasDpp);
}

TEST(DecodeDppModifiers, MissingImmModifierFatalError) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  const unsigned opc = llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp_gfx12;
  llvm::MCInst inst = makeTemplateInst(*state.instrInfo, opc);
  int bcIdx =
      llvm::AMDGPU::getNamedOperandIdx(opc, llvm::AMDGPU::OpName::bound_ctrl);
  ASSERT_GE(bcIdx, 0);
  inst.getOperand(static_cast<unsigned>(bcIdx)) =
      llvm::MCOperand::createReg(llvm::AMDGPU::VGPR0);

  const uint64_t ts = state.instrInfo->get(opc).TSFlags;
  transpiler::DecodedInst di = makeDi(std::move(inst), ts);

  ASSERT_DEATH(transpiler::decodeDppModifiers(di), "decodeDppModifiers");
}
