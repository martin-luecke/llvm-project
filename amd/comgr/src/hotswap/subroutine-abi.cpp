#include "subroutine-abi.h"

using namespace llvm;

namespace COMGR::hotswap {

void RegStateLayout::init(LLVMContext &C, unsigned NSgpr, unsigned NVgpr,
                          Type *ETy) {
  NumSgpr = NSgpr;
  NumVgpr = NVgpr;
  ExecTy = ETy;
  Type *I32Ty = Type::getInt32Ty(C);
  Ty = StructType::create(
      C,
      {ArrayType::get(I32Ty, NumSgpr), ArrayType::get(I32Ty, NumVgpr),
       I32Ty, I32Ty, ExecTy},
      "RegState");
}

Value *RegStateLayout::sgprGEP(IRBuilder<> &B, Value *Ptr,
                               unsigned Idx) const {
  return B.CreateStructGEP(Ty, Ptr, KSgprField,
                           "rs.sgpr." + Twine(Idx));
}

Value *RegStateLayout::vgprGEP(IRBuilder<> &B, Value *Ptr,
                               unsigned Idx) const {
  return B.CreateStructGEP(Ty, Ptr, KVgprField,
                           "rs.vgpr." + Twine(Idx));
}

Value *RegStateLayout::vccGEP(IRBuilder<> &B, Value *Ptr) const {
  return B.CreateStructGEP(Ty, Ptr, KVccField, "rs.vcc");
}

Value *RegStateLayout::sccGEP(IRBuilder<> &B, Value *Ptr) const {
  return B.CreateStructGEP(Ty, Ptr, KSccField, "rs.scc");
}

Value *RegStateLayout::execGEP(IRBuilder<> &B, Value *Ptr) const {
  return B.CreateStructGEP(Ty, Ptr, KExecField, "rs.exec");
}

void emitRegStateFlush(IRBuilder<> &B, const AllocaRegFile &Regs,
                       Value *StatePtr, const RegStateLayout &Layout,
                       ArrayRef<unsigned> VgprIndices,
                       ArrayRef<unsigned> SgprIndices) {
  Type *I32Ty = B.getInt32Ty();
  Value *SgprBase = B.CreateStructGEP(Layout.Ty, StatePtr,
                                      RegStateLayout::KSgprField);
  for (unsigned Idx : SgprIndices) {
    if (Idx >= Regs.Sgpr.size() || !Regs.Sgpr[Idx])
      continue;
    Value *ElemPtr =
        B.CreateInBoundsGEP(I32Ty, SgprBase, B.getInt32(Idx));
    Value *Val = B.CreateLoad(I32Ty, Regs.Sgpr[Idx]);
    B.CreateStore(Val, ElemPtr);
  }
  Value *VgprBase = B.CreateStructGEP(Layout.Ty, StatePtr,
                                      RegStateLayout::KVgprField);
  for (unsigned Idx : VgprIndices) {
    if (Idx >= Regs.Vgpr.size() || !Regs.Vgpr[Idx])
      continue;
    Value *ElemPtr =
        B.CreateInBoundsGEP(I32Ty, VgprBase, B.getInt32(Idx));
    Value *Val = B.CreateLoad(I32Ty, Regs.Vgpr[Idx]);
    B.CreateStore(Val, ElemPtr);
  }
}

void emitRegStateReload(IRBuilder<> &B, const AllocaRegFile &Regs,
                        Value *StatePtr, const RegStateLayout &Layout,
                        ArrayRef<unsigned> VgprIndices,
                        ArrayRef<unsigned> SgprIndices) {
  Type *I32Ty = B.getInt32Ty();
  Value *SgprBase = B.CreateStructGEP(Layout.Ty, StatePtr,
                                      RegStateLayout::KSgprField);
  for (unsigned Idx : SgprIndices) {
    if (Idx >= Regs.Sgpr.size() || !Regs.Sgpr[Idx])
      continue;
    Value *ElemPtr =
        B.CreateInBoundsGEP(I32Ty, SgprBase, B.getInt32(Idx));
    Value *Val = B.CreateLoad(I32Ty, ElemPtr);
    B.CreateStore(Val, Regs.Sgpr[Idx]);
  }
  Value *VgprBase = B.CreateStructGEP(Layout.Ty, StatePtr,
                                      RegStateLayout::KVgprField);
  for (unsigned Idx : VgprIndices) {
    if (Idx >= Regs.Vgpr.size() || !Regs.Vgpr[Idx])
      continue;
    Value *ElemPtr =
        B.CreateInBoundsGEP(I32Ty, VgprBase, B.getInt32(Idx));
    Value *Val = B.CreateLoad(I32Ty, ElemPtr);
    B.CreateStore(Val, Regs.Vgpr[Idx]);
  }
}

void emitSubroutinePrologue(IRBuilder<> &B, const AllocaRegFile &Regs,
                            Value *StatePtr, const RegStateLayout &Layout,
                            ArrayRef<unsigned> VgprIndices,
                            ArrayRef<unsigned> SgprIndices) {
  emitRegStateReload(B, Regs, StatePtr, Layout, VgprIndices, SgprIndices);
}

void emitSubroutineEpilogue(IRBuilder<> &B, const AllocaRegFile &Regs,
                            Value *StatePtr, const RegStateLayout &Layout,
                            ArrayRef<unsigned> VgprIndices,
                            ArrayRef<unsigned> SgprIndices) {
  emitRegStateFlush(B, Regs, StatePtr, Layout, VgprIndices, SgprIndices);
}

} // namespace COMGR::hotswap
