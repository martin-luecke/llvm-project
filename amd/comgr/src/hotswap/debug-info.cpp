//===- debug-info.cpp - DWARF preservation for hotswap raiser -----------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "debug-info.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/DebugInfo/DWARF/DWARFDie.h"
#include "llvm/DebugInfo/DWARF/DWARFUnit.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace COMGR::hotswap {

//===----------------------------------------------------------------------===//
// KernelDwarfSource
//===----------------------------------------------------------------------===//

KernelDwarfSource::KernelDwarfSource() = default;
KernelDwarfSource::~KernelDwarfSource() = default;

std::unique_ptr<KernelDwarfSource>
KernelDwarfSource::create(llvm::MemoryBufferRef CodeObject) {
  // Parse failures are silent -- "not an ELF" and "no DWARF" both mean
  // "skip debug-info emission".
  auto BinOrErr = llvm::object::createBinary(CodeObject);
  if (!BinOrErr) {
    llvm::consumeError(BinOrErr.takeError());
    return nullptr;
  }

  auto *Obj = llvm::dyn_cast<llvm::object::ObjectFile>(BinOrErr->get());
  if (!Obj)
    return nullptr;

  // With no .debug_info section the context has zero CUs and lookupLine
  // returns nullopt for every PC -- the outcome we want.
  auto Ctx = llvm::DWARFContext::create(*Obj);
  if (!Ctx)
    return nullptr;
  if (Ctx->getNumCompileUnits() == 0)
    return nullptr;

  std::unique_ptr<KernelDwarfSource> Src(new KernelDwarfSource());
  Src->OwnedBinary = std::move(*BinOrErr);
  Src->Ctx = std::move(Ctx);
  return Src;
}

std::optional<uint64_t>
KernelDwarfSource::kernelVA(llvm::StringRef KernelName) const {
  if (auto It = KernelVACache.find(KernelName); It != KernelVACache.end())
    return It->second;

  auto *Obj = llvm::dyn_cast<llvm::object::ObjectFile>(OwnedBinary.get());
  if (!Obj)
    return std::nullopt;

  for (const llvm::object::SymbolRef &Sym : Obj->symbols()) {
    llvm::Expected<llvm::StringRef> NameOrErr = Sym.getName();
    if (!NameOrErr) {
      llvm::consumeError(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != KernelName)
      continue;
    llvm::Expected<uint64_t> AddrOrErr = Sym.getAddress();
    if (!AddrOrErr) {
      llvm::consumeError(AddrOrErr.takeError());
      return std::nullopt;
    }
    CachedKernelNames.emplace_back(KernelName.str());
    llvm::StringRef Key(CachedKernelNames.back());
    KernelVACache.insert({Key, *AddrOrErr});
    return *AddrOrErr;
  }
  return std::nullopt;
}

std::optional<llvm::DILineInfo>
KernelDwarfSource::lookupLine(llvm::StringRef KernelName,
                              uint64_t LocalPC) const {
  std::optional<uint64_t> Base = kernelVA(KernelName);
  if (!Base)
    return std::nullopt;

  llvm::object::SectionedAddress Addr;
  Addr.Address = *Base + LocalPC;
  // UndefSection makes DWARFContext search all CUs by address range.
  Addr.SectionIndex = llvm::object::SectionedAddress::UndefSection;

  llvm::DILineInfoSpecifier Spec(
      llvm::DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,
      llvm::DILineInfoSpecifier::FunctionNameKind::LinkageName);
  std::optional<llvm::DILineInfo> LI =
      Ctx->getLineInfoForAddress(Addr, Spec);
  if (!LI || LI->Line == 0)
    return std::nullopt;
  return LI;
}

llvm::DWARFDie
KernelDwarfSource::findSubprogram(llvm::StringRef KernelName) const {
  std::optional<uint64_t> Base = kernelVA(KernelName);
  if (!Base)
    return llvm::DWARFDie();

  llvm::object::SectionedAddress Addr;
  Addr.Address = *Base;
  Addr.SectionIndex = llvm::object::SectionedAddress::UndefSection;
  llvm::DWARFCompileUnit *CU = Ctx->getCompileUnitForCodeAddress(Addr.Address);
  if (!CU)
    return llvm::DWARFDie();

  // Top-level DW_TAG_subprogram covering Base (we only need the outermost).
  llvm::DWARFDie CUDie = CU->getUnitDIE(/*ExtractUnitDIEOnly=*/false);
  if (!CUDie)
    return llvm::DWARFDie();
  for (llvm::DWARFDie Child : CUDie.children()) {
    if (Child.getTag() != llvm::dwarf::DW_TAG_subprogram)
      continue;
    if (Child.addressRangeContainsAddress(*Base))
      return Child;
  }
  return llvm::DWARFDie();
}

llvm::DWARFDie
KernelDwarfSource::findCompileUnit(llvm::StringRef KernelName) const {
  std::optional<uint64_t> Base = kernelVA(KernelName);
  if (!Base)
    return llvm::DWARFDie();
  llvm::DWARFCompileUnit *CU = Ctx->getCompileUnitForCodeAddress(*Base);
  if (!CU)
    return llvm::DWARFDie();
  return CU->getUnitDIE(/*ExtractUnitDIEOnly=*/true);
}

//===----------------------------------------------------------------------===//
// DebugInfoBuilder
//===----------------------------------------------------------------------===//

namespace {

/// String attribute off a DIE, or empty StringRef if absent.
llvm::StringRef dieString(const llvm::DWARFDie &Die,
                          llvm::dwarf::Attribute Attr) {
  if (!Die)
    return {};
  if (auto Val = Die.find(Attr)) {
    if (auto S = Val->getAsCString()) {
      if (*S)
        return llvm::StringRef(*S);
    }
  }
  return {};
}

unsigned dieUnsigned(const llvm::DWARFDie &Die, llvm::dwarf::Attribute Attr,
                     unsigned Default = 0) {
  if (!Die)
    return Default;
  if (auto Val = Die.find(Attr)) {
    if (auto U = Val->getAsUnsignedConstant())
      return static_cast<unsigned>(*U);
  }
  return Default;
}

} // namespace

DebugInfoBuilder::DebugInfoBuilder(llvm::Module &MIn,
                                   const KernelDwarfSource &SourceIn,
                                   std::string KernelNameIn)
    : M(MIn), Source(SourceIn), KernelName(std::move(KernelNameIn)),
      DIB(std::make_unique<llvm::DIBuilder>(MIn)) {}

DebugInfoBuilder::~DebugInfoBuilder() = default;

std::unique_ptr<DebugInfoBuilder>
DebugInfoBuilder::create(llvm::Module &M, const KernelDwarfSource &Source,
                         llvm::StringRef KernelName) {
  llvm::DWARFDie SubprogramDie = Source.findSubprogram(KernelName);
  if (!SubprogramDie)
    return nullptr;
  llvm::DWARFDie CUDie = Source.findCompileUnit(KernelName);
  if (!CUDie)
    return nullptr;

  std::unique_ptr<DebugInfoBuilder> B(
      new DebugInfoBuilder(M, Source, KernelName.str()));

  // Mirror the input CU's name + comp_dir + producer + language, falling
  // back to C99 if the input recorded no language.
  llvm::StringRef CUName = dieString(CUDie, llvm::dwarf::DW_AT_name);
  llvm::StringRef CompDir = dieString(CUDie, llvm::dwarf::DW_AT_comp_dir);
  llvm::StringRef Producer = dieString(CUDie, llvm::dwarf::DW_AT_producer);

  uint16_t LangTag = llvm::dwarf::DW_LANG_C99;
  if (auto L = CUDie.getLanguage())
    LangTag = static_cast<uint16_t>(*L);

  B->File = B->DIB->createFile(CUName.empty() ? llvm::StringRef(B->KernelName)
                                              : CUName,
                               CompDir);
  B->CU = B->DIB->createCompileUnit(
      llvm::DISourceLanguageName(LangTag), B->File,
      Producer.empty() ? llvm::StringRef("AMD Comgr hotswap raiser")
                       : Producer,
      /*isOptimized=*/true, /*Flags=*/"", /*RV=*/0);

  // Mirror subprogram name + linkage + decl line. Subroutine type is an
  // empty `void()` placeholder; full parameter type reconstruction is
  // Phase 2.
  llvm::StringRef SubName = dieString(SubprogramDie, llvm::dwarf::DW_AT_name);
  llvm::StringRef LinkageName =
      dieString(SubprogramDie, llvm::dwarf::DW_AT_linkage_name);
  unsigned LineNo = dieUnsigned(SubprogramDie, llvm::dwarf::DW_AT_decl_line, 0);
  unsigned ScopeLine =
      dieUnsigned(SubprogramDie, llvm::dwarf::DW_AT_decl_line, LineNo);

  llvm::DITypeArray EmptyTypeArray = B->DIB->getOrCreateTypeArray({});
  llvm::DISubroutineType *SubTy = B->DIB->createSubroutineType(EmptyTypeArray);

  B->SP = B->DIB->createFunction(
      B->CU, SubName.empty() ? llvm::StringRef(KernelName) : SubName,
      LinkageName, B->File, LineNo, SubTy, ScopeLine,
      llvm::DINode::FlagPrototyped,
      llvm::DISubprogram::SPFlagDefinition |
          llvm::DISubprogram::SPFlagOptimized);

  return B;
}

void DebugInfoBuilder::attachSubprogramTo(llvm::Function &Kernel) {
  if (SP)
    Kernel.setSubprogram(SP);
}

llvm::DILocation *DebugInfoBuilder::locationFor(uint64_t LocalPC) const {
  if (!SP)
    return nullptr;
  std::optional<llvm::DILineInfo> LI = Source.lookupLine(KernelName, LocalPC);
  if (!LI || LI->Line == 0)
    return nullptr;
  // Single DILocation scoped under the kernel subprogram; inline-chain
  // reconstruction is Phase 2.
  return llvm::DILocation::get(M.getContext(), LI->Line, LI->Column, SP);
}

void DebugInfoBuilder::finalize() {
  if (!SP)
    return;

  // Emit DILocalVariable shells for each source variable / parameter.
  // These are unbound (no dbg.declare), so debuggers show them as
  // "optimized out"; binding to storage is Phase 2.
  llvm::DWARFDie SubprogramDie = Source.findSubprogram(KernelName);
  if (SubprogramDie) {
    auto *VoidTy = DIB->createUnspecifiedType("void");
    unsigned ParamIdx = 1;
    for (llvm::DWARFDie Child : SubprogramDie.children()) {
      llvm::dwarf::Tag Tag = Child.getTag();
      if (Tag != llvm::dwarf::DW_TAG_formal_parameter &&
          Tag != llvm::dwarf::DW_TAG_variable)
        continue;
      llvm::StringRef Name = dieString(Child, llvm::dwarf::DW_AT_name);
      if (Name.empty())
        continue;
      unsigned Line =
          dieUnsigned(Child, llvm::dwarf::DW_AT_decl_line, 0);
      if (Tag == llvm::dwarf::DW_TAG_formal_parameter) {
        DIB->createParameterVariable(SP, Name, ParamIdx++, File, Line, VoidTy,
                                     /*AlwaysPreserve=*/true);
      } else {
        DIB->createAutoVariable(SP, Name, File, Line, VoidTy,
                                /*AlwaysPreserve=*/true);
      }
    }
  }

  DIB->finalize();

  // Guard the flags for idempotence (the raiser's module is fresh).
  if (!M.getModuleFlag("Debug Info Version"))
    M.addModuleFlag(llvm::Module::Warning, "Debug Info Version",
                    llvm::DEBUG_METADATA_VERSION);
  if (!M.getModuleFlag("Dwarf Version"))
    M.addModuleFlag(llvm::Module::Max, "Dwarf Version", DwarfVersion);
}

} // namespace COMGR::hotswap
