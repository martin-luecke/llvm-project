#include "handle_valu_output_mods.hpp"

#include "canonical_op.hpp"

#include "llvm/ADT/Twine.h"
#include "Utils/AMDGPUBaseInfo.h"

using namespace llvm;

namespace transpiler {

namespace {

bool readNamedImm(const DecodedInst &di, AMDGPU::OpName name, int64_t &out) {
  int idx = AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), name);
  if (idx < 0 || static_cast<unsigned>(idx) >= di.inst.getNumOperands())
    return false;
  const MCOperand &op = di.inst.getOperand(static_cast<unsigned>(idx));
  if (!op.isImm())
    return false;
  out = op.getImm();
  return true;
}

std::optional<int64_t> readNamedImmOperand(const DecodedInst &di,
                                           AMDGPU::OpName name) {
  int64_t value = 0;
  if (!readNamedImm(di, name, value))
    return std::nullopt;
  return value;
}

} // namespace

bool requireDefaultVOP3OutputMods(const DecodedInst &di, HandlerResult &hr,
                                   StringRef diagnosticName,
                                   VOP3OutputModPresence presence,
                                   VOP3OutputModDiag diag) {
  const int clampIdx =
      AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), AMDGPU::OpName::clamp);
  const int omodIdx =
      AMDGPU::getNamedOperandIdx(di.inst.getOpcode(), AMDGPU::OpName::omod);

  if (presence == VOP3OutputModPresence::IfPresent) {
    if (clampIdx < 0 && omodIdx < 0)
      return true;

    int64_t clamp = 0;
    int64_t omod = 0;
    if ((clampIdx >= 0 && !readNamedImm(di, AMDGPU::OpName::clamp, clamp)) ||
        (omodIdx >= 0 && !readNamedImm(di, AMDGPU::OpName::omod, omod))) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          (Twine(diagnosticName) +
           " has malformed clamp/omod operands; operand table layout does not "
           "match the expected VOP3 profile")
              .str());
      return false;
    }

    if (clamp != 0 || omod != 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          (Twine(diagnosticName) +
           " with non-default clamp/omod is not yet lifted; output modifier "
           "semantics must not be silently dropped")
              .str());
      return false;
    }
    return true;
  }

  // Required presence: both operands must exist and be immediates.
  if (diag == VOP3OutputModDiag::FpValuSplit) {
    std::optional<int64_t> clamp =
        readNamedImmOperand(di, AMDGPU::OpName::clamp);
    if (!clamp) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          (Twine(diagnosticName) +
           " missing immediate clamp operand; operand table layout does not "
           "match the expected VOP3 profile")
              .str());
      return false;
    }
    if (*clamp != 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          (Twine(diagnosticName) +
           " has clamp=1; VOP3 floating-point output clamp is not modeled for "
           "this opcode")
              .str());
      return false;
    }

    std::optional<int64_t> omod = readNamedImmOperand(di, AMDGPU::OpName::omod);
    if (!omod) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          (Twine(diagnosticName) +
           " missing immediate omod operand; operand table layout does not "
           "match the expected VOP3 profile")
              .str());
      return false;
    }
    if (*omod != 0) {
      hr.failure = RaiseFailure::unsupportedShape(
          di, "VOP3",
          (Twine(diagnosticName) +
           " has nonzero omod; VOP3 floating-point output scaling is not "
           "modeled for this opcode")
              .str());
      return false;
    }
    return true;
  }

  int64_t clamp = 0;
  int64_t omod = 0;
  if (!readNamedImm(di, AMDGPU::OpName::clamp, clamp) ||
      !readNamedImm(di, AMDGPU::OpName::omod, omod)) {
    const char *missingSuffix =
        diag == VOP3OutputModDiag::PseudoScalar
            ? " missing immediate clamp/omod operands; operand table layout "
              "does not match the gfx12 VOP3 pseudo-scalar profile"
            : " missing immediate clamp/omod operands; operand table layout "
              "does not match the expected VOP3 profile";
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3", (Twine(diagnosticName) + missingSuffix).str());
    return false;
  }

  if (clamp != 0 || omod != 0) {
    const char *refusalSuffix =
        diag == VOP3OutputModDiag::PseudoScalar
            ? " with non-default clamp/omod is not yet lifted; the base "
              "instruction is supported through an AMDGPU hardware intrinsic, "
              "but output modifier semantics must not be silently dropped"
            : " with non-default clamp/omod is not yet lifted; output modifier "
              "semantics must not be silently dropped";
    hr.failure = RaiseFailure::unsupportedShape(
        di, "VOP3", (Twine(diagnosticName) + refusalSuffix).str());
    return false;
  }

  return true;
}

} // namespace transpiler
