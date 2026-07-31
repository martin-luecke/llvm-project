//===- atomic-replica.cpp - scaled-dispatch atomic gating -----------------===//

#include "atomic-replica.h"

#include "canonical-op.h"
#include "decoded-inst.h"
#include "raise-context.h"
#include "raise-failure.h"

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"

using namespace llvm;

namespace COMGR::hotswap {

namespace {

// An atomic RMW is idempotent when re-applying the same operand to the same
// location is a no-op, so a scaled-dispatch replica issuing it a second time
// leaves memory unchanged: the bitwise and/or and the integer/FP min/max
// family. Everything else (add/sub/fadd/xor/swap/cmpswap/inc/dec/...) changes
// memory on each issue and must be gated to one replica. Classify by the
// canonical op so the FLAT/GLOBAL, MUBUF, and DS handlers share one table;
// anything not listed here is treated as non-idempotent (the safe default:
// over-gating an idempotent op is still correct, under-gating miscomputes).
bool isIdempotentAtomic(CanonicalOp Op) {
  switch (Op) {
  case CanonicalOp::FLAT_ATOMIC_AND:
  case CanonicalOp::FLAT_ATOMIC_OR:
  case CanonicalOp::FLAT_ATOMIC_SMIN:
  case CanonicalOp::FLAT_ATOMIC_SMAX:
  case CanonicalOp::FLAT_ATOMIC_UMIN:
  case CanonicalOp::FLAT_ATOMIC_UMAX:
  case CanonicalOp::FLAT_ATOMIC_AND_X2:
  case CanonicalOp::FLAT_ATOMIC_OR_X2:
  case CanonicalOp::FLAT_ATOMIC_SMIN_X2:
  case CanonicalOp::FLAT_ATOMIC_SMAX_X2:
  case CanonicalOp::FLAT_ATOMIC_UMIN_X2:
  case CanonicalOp::FLAT_ATOMIC_UMAX_X2:
  case CanonicalOp::FLAT_ATOMIC_MIN_NUM_F64:
  case CanonicalOp::FLAT_ATOMIC_MAX_NUM_F64:
  case CanonicalOp::GLOBAL_ATOMIC_AND:
  case CanonicalOp::GLOBAL_ATOMIC_OR:
  case CanonicalOp::GLOBAL_ATOMIC_SMIN:
  case CanonicalOp::GLOBAL_ATOMIC_SMAX:
  case CanonicalOp::GLOBAL_ATOMIC_UMIN:
  case CanonicalOp::GLOBAL_ATOMIC_UMAX:
  case CanonicalOp::GLOBAL_ATOMIC_MIN_NUM_F64:
  case CanonicalOp::GLOBAL_ATOMIC_MAX_NUM_F64:
  case CanonicalOp::BUFFER_ATOMIC_AND:
  case CanonicalOp::BUFFER_ATOMIC_OR:
  case CanonicalOp::BUFFER_ATOMIC_MIN_F64:
  case CanonicalOp::BUFFER_ATOMIC_MAX_F64:
  case CanonicalOp::BUFFER_ATOMIC_MIN_NUM_F64:
  case CanonicalOp::BUFFER_ATOMIC_MAX_NUM_F64:
    return true;
  default:
    return false;
  }
}

} // namespace

Expected<bool> needsOneReplicaGate(RaiseContext &Ctx, const DecodedInst &Di,
                                   StringRef Format) {
  if (!Ctx.Projection.usesScaledDispatch())
    return false;
  if (Di.NumDefs > 0)
    return RaiseFailure::unsupportedInstructionForm(
        Di, Format,
        "returning atomic RMW under a scaled dispatch: the source lane and its "
        "active replica each issue the RMW and read a different value, which "
        "cannot be reconciled without a replica broadcast; refuse rather than "
        "miscompute");
  return !isIdempotentAtomic(Di.CanonOp);
}

void emitAtomicUnderOneReplica(RaiseContext &Ctx, function_ref<void()> Emit) {
  Value *LaneId = Ctx.emitLaneIdx();
  Value *WsC = ConstantInt::get(LaneId->getType(), Ctx.Isa.WaveSize);
  Value *IsRep0 = Ctx.B.CreateICmpULT(LaneId, WsC, "one_replica");
  BasicBlock *PreBb = Ctx.B.GetInsertBlock();
  Function *Fn = PreBb->getParent();
  BasicBlock *DoBb = BasicBlock::Create(Ctx.C, "atomic_do", Fn);
  BasicBlock *SkipBb = BasicBlock::Create(Ctx.C, "atomic_skip", Fn);
  Ctx.B.CreateCondBr(IsRep0, DoBb, SkipBb);
  Ctx.B.SetInsertPoint(DoBb);
  Emit();
  Ctx.B.CreateBr(SkipBb);
  Ctx.B.SetInsertPoint(SkipBb);
  // The manual EXEC-narrowing branch invalidates the memoised lane-active bit
  // for any later emission of this instruction.
  Ctx.resetLaneActiveCache();
}

} // namespace COMGR::hotswap
