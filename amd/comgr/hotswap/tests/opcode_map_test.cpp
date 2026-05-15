#include "../mc_state.hpp"
#include "../opcode_map.hpp"

#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/Support/TargetSelect.h"

#include "gtest/gtest.h"

#include <mutex>

namespace {

void ensureAMDGPURegistered() {
  static std::once_flag flag;
  std::call_once(flag, []() {
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
  transpiler::OpcodeMap map;
  EXPECT_EQ(map.lookup(0), transpiler::CanonicalOp::Unknown);
  EXPECT_EQ(map.lookup(12345), transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, BuildOnGfx942IsBenign) {
  ensureAMDGPURegistered();
  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx942"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  // No handler patches have landed yet, so every MC opcode resolves to
  // `Unknown` — the raiser bails on the first decoded instruction.
  EXPECT_EQ(map.lookup(0), transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250AddMinRealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_ADD_MIN_U32_e64_gfx1250),
            transpiler::CanonicalOp::V_ADD_MIN_U32);
}

TEST(OpcodeMap, Gfx1250SubNcU16RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_gfx12),
            transpiler::CanonicalOp::V_SUB_NC_U16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_SUB_NC_U16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_SUB_NC_U16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp_gfx12),
            transpiler::CanonicalOp::V_SUB_NC_U16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_SUB_NC_U16_fake16_e64_dpp8_gfx12),
            transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250ScalarF16ToF32RealOpcodesMapToCanonicalOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::S_CVT_F16_F32_gfx12),
            transpiler::CanonicalOp::S_CVT_F16_F32);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::S_CVT_F32_F16_gfx12),
            transpiler::CanonicalOp::S_CVT_F32_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::S_CVT_HI_F32_F16_gfx12),
            transpiler::CanonicalOp::S_CVT_HI_F32_F16);
}

TEST(OpcodeMap, Gfx1250VectorF32F64RealOpcodesMapToCanonicalOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_CVT_F32_F64_e64_gfx12),
            transpiler::CanonicalOp::V_CVT_F32_F64);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_CVT_F64_F32_e64_gfx12),
            transpiler::CanonicalOp::V_CVT_F64_F32);
}

TEST(OpcodeMap, Gfx1250AddSubNcI16RealOpcodesMapToSemOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(
      map.lookup(llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_fake16_e64_gfx12),
      transpiler::CanonicalOp::V_ADD_NC_I16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_ADD_NC_I16);
  EXPECT_EQ(
      map.lookup(llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_fake16_e64_dpp_gfx12),
      transpiler::CanonicalOp::V_ADD_NC_I16);
  EXPECT_EQ(
      map.lookup(llvm::AMDGPU::V_ADD_NC_I16V_ADD_I16_fake16_e64_dpp8_gfx12),
      transpiler::CanonicalOp::Unknown);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_fake16_e64_gfx12),
            transpiler::CanonicalOp::V_SUB_NC_I16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_SUB_NC_I16);
  EXPECT_EQ(
      map.lookup(llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_fake16_e64_dpp_gfx12),
      transpiler::CanonicalOp::V_SUB_NC_I16);
  EXPECT_EQ(
      map.lookup(llvm::AMDGPU::V_SUB_NC_I16V_SUB_I16_fake16_e64_dpp8_gfx12),
      transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250ScalarF32RoundingOpcodesMapToCanonicalOps) {
  ensureAMDGPURegistered();

  transpiler::MCState State;
  ASSERT_TRUE(transpiler::initMCState(State, "gfx1250"));

  transpiler::OpcodeMap Map;
  Map.build(*State.instrInfo);

  EXPECT_EQ(Map.lookup(llvm::AMDGPU::S_CEIL_F32),
            transpiler::CanonicalOp::S_CEIL_F32);
  EXPECT_EQ(Map.lookup(llvm::AMDGPU::S_FLOOR_F32),
            transpiler::CanonicalOp::S_FLOOR_F32);
  EXPECT_EQ(Map.lookup(llvm::AMDGPU::S_TRUNC_F32),
            transpiler::CanonicalOp::S_TRUNC_F32);
  EXPECT_EQ(Map.lookup(llvm::AMDGPU::S_RNDNE_F32),
            transpiler::CanonicalOp::S_RNDNE_F32);
}

TEST(OpcodeMap, Gfx1250Min3RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MIN3_U32_e64_gfx12),
            transpiler::CanonicalOp::V_MIN3_U32);
}

TEST(OpcodeMap, Gfx1250Dot4I32IU8RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8),
            transpiler::CanonicalOp::V_DOT4_I32_IU8);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_DOT4_I32_IU8_gfx12),
            transpiler::CanonicalOp::V_DOT4_I32_IU8);
}

TEST(OpcodeMap, Gfx1250PkFmaF16RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_FMA_F16_gfx12),
            transpiler::CanonicalOp::V_PK_FMA_F16);
}

TEST(OpcodeMap, Gfx1250PkAddBF16RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_ADD_BF16_gfx1250),
            transpiler::CanonicalOp::V_PK_ADD_BF16);
}

TEST(OpcodeMap, Gfx1250PkFmaBF16RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_FMA_BF16_gfx1250),
            transpiler::CanonicalOp::V_PK_FMA_BF16);
}

TEST(OpcodeMap, Gfx1250PkBF16SiblingsRealOpcodesMapToSemOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_MUL_BF16_gfx1250),
            transpiler::CanonicalOp::V_PK_MUL_BF16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_MIN_NUM_BF16_gfx1250),
            transpiler::CanonicalOp::V_PK_MIN_NUM_BF16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_PK_MAX_NUM_BF16_gfx1250),
            transpiler::CanonicalOp::V_PK_MAX_NUM_BF16);
}

TEST(OpcodeMap, Gfx1250FmaMixF16HalfResultRealOpcodesMapToSemOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXLO_F16_gfx12),
            transpiler::CanonicalOp::V_FMA_MIXLO_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXHI_F16_gfx12),
            transpiler::CanonicalOp::V_FMA_MIXHI_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXLO_F16_dpp_gfx12),
            transpiler::CanonicalOp::V_FMA_MIXLO_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXHI_F16_dpp8_gfx12),
            transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250FmaMixBF16HalfResultRealOpcodesMapToSemOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXLO_BF16_gfx1250),
            transpiler::CanonicalOp::V_FMA_MIXLO_BF16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXHI_BF16_gfx1250),
            transpiler::CanonicalOp::V_FMA_MIXHI_BF16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXLO_BF16_dpp_gfx1250),
            transpiler::CanonicalOp::V_FMA_MIXLO_BF16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_FMA_MIXHI_BF16_dpp8_gfx1250),
            transpiler::CanonicalOp::Unknown);
}

TEST(OpcodeMap, Gfx1250MadI32I24RealOpcodeMapsToSemOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAD_I32_I24_e64_gfx12),
            transpiler::CanonicalOp::V_MAD_I32_I24);
}

TEST(OpcodeMap, Gfx1250CvtScalef32Pk8Fp8F32RealOpcodeMapsToCanonicalOp) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  // V_CVT_SCALEF32_PK8_FP8_F32: gfx1250-only packed-8 scaled FP8
  // conversion (VOP3 opcode 0x2c3, profile VOP_V2I32_V8F32_F32 in
  // VOP3Instructions.td:1883).
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_CVT_SCALEF32_PK8_FP8_F32_e64_gfx1250),
            transpiler::CanonicalOp::V_CVT_SCALEF32_PK8_FP8_F32);
}

TEST(OpcodeMap, Gfx1250Maximum3Minimum3F32RealOpcodesMapToCanonicalOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  // V_MAXIMUM3_F32 / V_MINIMUM3_F32: gfx11+/gfx12 ternary IEEE-754
  // NaN-propagating max/min.  HasMinimum3Maximum3F32 in AMDGPU.td:194.
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXIMUM3_F32_e64_gfx12),
            transpiler::CanonicalOp::V_MAXIMUM3_F32);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MINIMUM3_F32_e64_gfx12),
            transpiler::CanonicalOp::V_MINIMUM3_F32);
}

TEST(OpcodeMap, Gfx1250MaximumMinimumF32RealOpcodesMapToCanonicalOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  // V_MAXIMUMMINIMUM_F32 / V_MINIMUMMAXIMUM_F32: gfx11+/gfx12 ternary
  // IEEE-754 NaN-propagating clamp pair at VOP3 opcodes 0x26d / 0x26c.
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXIMUMMINIMUM_F32_e64_gfx12),
            transpiler::CanonicalOp::V_MAXIMUMMINIMUM_F32);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MINIMUMMAXIMUM_F32_e64_gfx12),
            transpiler::CanonicalOp::V_MINIMUMMAXIMUM_F32);
}

TEST(OpcodeMap, Gfx1250RelatedMinimumMaximumOpcodesMapToCanonicalOps) {
  ensureAMDGPURegistered();

  transpiler::MCState state;
  ASSERT_TRUE(transpiler::initMCState(state, "gfx1250"));

  transpiler::OpcodeMap map;
  map.build(*state.instrInfo);

  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXMIN_F32_e64),
            transpiler::CanonicalOp::V_MAXMIN_NUM_F32);
  // LLVM names the t16 .NUM f16 real opcodes with the gfx11 suffix in this
  // build; OpcodeMap still builds against gfx1250 MC state and validates the
  // alias collapse from that real form to the shared canonical pseudo.
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MINMAX_F16_t16_e64_gfx11),
            transpiler::CanonicalOp::V_MINMAX_NUM_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXMIN_F16_t16_e64_gfx11),
            transpiler::CanonicalOp::V_MAXMIN_NUM_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXIMUM_F16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_MAXIMUM_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MINIMUM_F16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_MINIMUM_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXIMUM3_F16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_MAXIMUM3_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MINIMUM3_F16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_MINIMUM3_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MAXIMUMMINIMUM_F16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_MAXIMUMMINIMUM_F16);
  EXPECT_EQ(map.lookup(llvm::AMDGPU::V_MINIMUMMAXIMUM_F16_t16_e64_gfx12),
            transpiler::CanonicalOp::V_MINIMUMMAXIMUM_F16);
}
