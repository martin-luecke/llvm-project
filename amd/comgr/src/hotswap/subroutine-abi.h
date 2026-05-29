#ifndef COMGR_HOTSWAP_SUBROUTINE_ABI_H
#define COMGR_HOTSWAP_SUBROUTINE_ABI_H

#include "reg-file.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"

#include <vector>

namespace COMGR::hotswap {

// Field layout of the register-state struct passed between the kernel
// and subroutine functions. Each field is an array of i32 (or i1 for
// flags). The struct lives in private address space (addrspace 5,
// scratch memory) and is allocated by the kernel.
//
//   { [NumSgpr x i32], [NumVgpr x i32], i32 vcc, i32 scc, ExecTy exec }
//
// The VGPR/SGPR arrays mirror AllocaRegFile's index space so that
// flush/reload can use the same indices the handlers use.
struct RegStateLayout {
  static constexpr unsigned KSgprField = 0;
  static constexpr unsigned KVgprField = 1;
  static constexpr unsigned KVccField = 2;
  static constexpr unsigned KSccField = 3;
  static constexpr unsigned KExecField = 4;

  unsigned NumSgpr;
  unsigned NumVgpr;
  llvm::Type *ExecTy;
  llvm::StructType *Ty = nullptr;

  void init(llvm::LLVMContext &C, unsigned NumSgpr, unsigned NumVgpr,
            llvm::Type *ExecTy);

  llvm::Value *sgprGEP(llvm::IRBuilder<> &B, llvm::Value *Ptr,
                       unsigned Idx) const;
  llvm::Value *vgprGEP(llvm::IRBuilder<> &B, llvm::Value *Ptr,
                       unsigned Idx) const;
  llvm::Value *vccGEP(llvm::IRBuilder<> &B, llvm::Value *Ptr) const;
  llvm::Value *sccGEP(llvm::IRBuilder<> &B, llvm::Value *Ptr) const;
  llvm::Value *execGEP(llvm::IRBuilder<> &B, llvm::Value *Ptr) const;
};

// Flush a subset of registers from AllocaRegFile allocas into the
// RegState struct. Called before a subroutine call.
void emitRegStateFlush(llvm::IRBuilder<> &B, const AllocaRegFile &Regs,
                       llvm::Value *StatePtr, const RegStateLayout &Layout,
                       llvm::ArrayRef<unsigned> VgprIndices,
                       llvm::ArrayRef<unsigned> SgprIndices);

// Reload a subset of registers from the RegState struct back into
// AllocaRegFile allocas. Called after a subroutine call returns.
void emitRegStateReload(llvm::IRBuilder<> &B, const AllocaRegFile &Regs,
                        llvm::Value *StatePtr, const RegStateLayout &Layout,
                        llvm::ArrayRef<unsigned> VgprIndices,
                        llvm::ArrayRef<unsigned> SgprIndices);

// Initialize a subroutine's local AllocaRegFile from the RegState
// struct pointer (prologue). Loads all registers in the given index
// sets from the struct into the allocas.
void emitSubroutinePrologue(llvm::IRBuilder<> &B, const AllocaRegFile &Regs,
                            llvm::Value *StatePtr,
                            const RegStateLayout &Layout,
                            llvm::ArrayRef<unsigned> VgprIndices,
                            llvm::ArrayRef<unsigned> SgprIndices);

// Store a subroutine's local AllocaRegFile back to the RegState
// struct (epilogue). Stores all registers in the given index sets
// from the allocas into the struct. Called before ret void.
void emitSubroutineEpilogue(llvm::IRBuilder<> &B, const AllocaRegFile &Regs,
                            llvm::Value *StatePtr,
                            const RegStateLayout &Layout,
                            llvm::ArrayRef<unsigned> VgprIndices,
                            llvm::ArrayRef<unsigned> SgprIndices);

} // namespace COMGR::hotswap

#endif
