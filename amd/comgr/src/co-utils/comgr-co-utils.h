//===- comgr-co-utils.h - Shared code-object utilities ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared low-level code-object utilities for the comgr stack:
// ELF parsing, AMDGPU MsgPack metadata extraction, and ISA-name handling
// derived from the ELF header.
//
// Header-only so it is usable by every layer without inverting the existing
// link order: `amd_comgr` (`comgr-metadata.cpp`), the hotswap binary-patching
// path (`comgr-hotswap-*.cpp`), and the hotswap transpiler subproject
// (`src/hotswap/code_object_utils.cpp`) all `#include` this directly. The
// only dependency is LLVM (`Object`, `BinaryFormat`, `Support`); no
// `DataObject` / `DataMeta` / `amd_comgr.h` coupling here.
//
// This is the seed of a planned `comgr-co-utils` shared module that will
// eventually also absorb the MC-stack setup duplicated across
// `comgr-disassembly.cpp`, `comgr-hotswap-llvm.cpp`, and `mc_state.cpp`.
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_CO_UTILS_H
#define COMGR_CO_UTILS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <memory>
#include <string>

namespace COMGR::co_utils {

// PAL metadata note type — same constant comgr-metadata.cpp uses;
// duplicated here so this header is self-contained.
inline constexpr uint32_t PalMetadataNoteType = 13;

// Merge `From` into `To`, copying nodes into `DestDoc`'s arena. Used to
// combine multiple PAL/AMDGPU metadata note records into a single root.
// Lifted from comgr-metadata.cpp so this header is self-contained.
inline bool mergeNoteRecords(llvm::msgpack::DocNode &From,
                             llvm::msgpack::DocNode &To,
                             llvm::StringRef VersionStrKey,
                             llvm::StringRef PrintfStrKey,
                             llvm::StringRef KernelStrKey,
                             llvm::msgpack::Document &DestDoc) {
  if (!From.isMap()) {
    return false;
  }
  if (To.isEmpty()) {
    To = DestDoc.copyNode(From);
    return true;
  }
  assert(To.isMap());

  // `printf` records must agree across notes if both define it.
  if (From.getMap().find(PrintfStrKey) != From.getMap().end()) {
    if (To.getMap().find(PrintfStrKey) != To.getMap().end()) {
      if (From.getMap()[PrintfStrKey] != To.getMap()[PrintfStrKey])
        return false;
    } else {
      To.getMap()[PrintfStrKey] = DestDoc.copyNode(From.getMap()[PrintfStrKey]);
    }
  }

  // `version` must agree if both define it.
  if (From.getMap().find(VersionStrKey) != From.getMap().end()) {
    if (To.getMap().find(VersionStrKey) != To.getMap().end()) {
      if (From.getMap()[VersionStrKey] != To.getMap()[VersionStrKey])
        return false;
    } else {
      To.getMap()[VersionStrKey] =
          DestDoc.copyNode(From.getMap()[VersionStrKey]);
    }
  }

  // `kernels` arrays from each note are concatenated.
  if (From.getMap().find(KernelStrKey) != From.getMap().end()) {
    if (!From.getMap()[KernelStrKey].isArray())
      return false;
    if (To.getMap().find(KernelStrKey) == To.getMap().end()) {
      To.getMap()[KernelStrKey] = DestDoc.getArrayNode();
    }
    if (!To.getMap()[KernelStrKey].isArray())
      return false;
    for (auto &K : From.getMap()[KernelStrKey].getArray())
      To.getMap()[KernelStrKey].getArray().push_back(DestDoc.copyNode(K));
  }
  return true;
}

// Process one ELF note. Recognises `NT_AMD_HSA_METADATA` (older YAML
// dialect, name="AMD") and `NT_AMDGPU_METADATA` (MsgPack, name="AMDGPU"),
// plus the PAL metadata note (type 13, name="AMD" or "AMDGPU"). Merges
// successive PAL/AMDGPU notes; the YAML-formatted HSA note may appear
// only once.
template <class ELFT>
inline bool
processElfNote(const typename llvm::object::ELFFile<ELFT>::Elf_Note &Note,
               llvm::msgpack::Document &Doc, llvm::msgpack::DocNode &Root,
               bool &EmitIntegerBooleans) {
  llvm::StringRef DescString = Note.getDescAsStringRef(4);

  if (Note.getName() == "AMD" &&
      Note.getType() == llvm::ELF::NT_AMD_HSA_METADATA) {
    if (!Root.isEmpty())
      return false;
    EmitIntegerBooleans = false;
    if (!Doc.fromYAML(DescString))
      return false;
    Root = Doc.getRoot();
    return true;
  }

  bool IsPal = (Note.getName() == "AMD" || Note.getName() == "AMDGPU") &&
               Note.getType() == PalMetadataNoteType;
  bool IsAmdgpu = Note.getName() == "AMDGPU" &&
                  Note.getType() == llvm::ELF::NT_AMDGPU_METADATA;
  if (IsPal || IsAmdgpu) {
    if (!Root.isEmpty() && EmitIntegerBooleans != true)
      return false;
    EmitIntegerBooleans = true;

    llvm::msgpack::Document TempDoc;
    if (!TempDoc.readFromBlob(DescString, false))
      return false;
    return mergeNoteRecords(TempDoc.getRoot(), Root, "amdhsa.version",
                            "amdhsa.printf", "amdhsa.kernels", Doc);
  }
  return false;
}

// Walk an ELF object's PT_NOTE program headers AND SHT_NOTE sections,
// merging every recognised AMDGPU metadata note into `Doc`. Program
// headers are tried first (matches comgr-metadata.cpp's preference);
// sections are the fallback for inputs that lack PT_NOTE entries.
//
// Returns `true` if at least one recognised note was processed and `Doc`
// holds the parsed root; returns `false` if no AMDGPU metadata note was
// present (caller decides whether that's an error). Returns an `Error`
// for ELF-parse failures.
template <class ELFT>
inline llvm::Expected<bool>
walkElfMetadata(const llvm::object::ELFObjectFile<ELFT> &Obj,
                llvm::msgpack::Document &Doc, bool &EmitIntegerBooleans) {
  bool Found = false;
  llvm::msgpack::DocNode Root;
  const llvm::object::ELFFile<ELFT> &ELFFile = Obj.getELFFile();

  // Program headers (preferred).
  auto PhdrsOrErr = ELFFile.program_headers();
  if (!PhdrsOrErr)
    return PhdrsOrErr.takeError();
  for (const auto &Phdr : *PhdrsOrErr) {
    if (Phdr.p_type != llvm::ELF::PT_NOTE)
      continue;
    llvm::Error Err = llvm::Error::success();
    for (const auto &Note : ELFFile.notes(Phdr, Err)) {
      if (processElfNote<ELFT>(Note, Doc, Root, EmitIntegerBooleans))
        Found = true;
    }
    if (Err)
      return std::move(Err);
  }

  if (!Found) {
    // Section-header fallback.
    auto SectionsOrErr = ELFFile.sections();
    if (!SectionsOrErr)
      return SectionsOrErr.takeError();
    for (const auto &Shdr : *SectionsOrErr) {
      if (Shdr.sh_type != llvm::ELF::SHT_NOTE)
        continue;
      llvm::Error Err = llvm::Error::success();
      for (const auto &Note : ELFFile.notes(Shdr, Err)) {
        if (processElfNote<ELFT>(Note, Doc, Root, EmitIntegerBooleans))
          Found = true;
      }
      if (Err)
        return std::move(Err);
    }
  }

  if (Found)
    Doc.getRoot() = Root;
  return Found;
}

// `MemoryBufferRef`-friendly metadata extractor that populates the
// caller-supplied `Doc`. Supports all 4 ELF endian/bit variants and walks
// both PT_NOTE program headers and SHT_NOTE sections. Returns `true` if at
// least one recognised AMDGPU metadata note was processed; `false` otherwise.
// `EmitIntegerBooleans` is set to `true` when the matched note used the
// MsgPack-encoded format (NT_AMDGPU_METADATA / PAL) and to `false` for the
// older YAML-encoded NT_AMD_HSA_METADATA format. Downstream consumers
// (`comgr.cpp`'s `iterate_map_metadata`) use this to decide whether 0/1
// integer values should be reported as booleans.
inline llvm::Expected<bool>
walkElfMetadataIntoDoc(llvm::MemoryBufferRef MB, llvm::msgpack::Document &Doc,
                       bool &EmitIntegerBooleans) {
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(MB);
  if (!ObjOrErr)
    return ObjOrErr.takeError();

  auto Walk = [&](auto &E) -> llvm::Expected<bool> {
    return walkElfMetadata(E, Doc, EmitIntegerBooleans);
  };

  auto *Base = ObjOrErr->get();
  if (auto *E = llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(Base))
    return Walk(*E);
  if (auto *E = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(Base))
    return Walk(*E);
  if (auto *E = llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(Base))
    return Walk(*E);
  if (auto *E = llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(Base))
    return Walk(*E);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported ELF variant");
}

// `MemoryBufferRef`-friendly metadata extractor that allocates a new
// `msgpack::Document`. Returns an error if no AMDGPU metadata note is
// present.
//
// The result is wrapped in a `unique_ptr` because `msgpack::Document`
// stores self-referencing `KindAndDocs[]` pointers in its constructor;
// moving the Document leaves those pointers dangling, so we keep the
// instance pinned on the heap.
inline llvm::Expected<std::unique_ptr<llvm::msgpack::Document>>
getElfMetadataDocument(llvm::MemoryBufferRef MB) {
  auto Doc = std::make_unique<llvm::msgpack::Document>();
  bool EmitIntegerBooleans = false;
  auto FoundOrErr = walkElfMetadataIntoDoc(MB, *Doc, EmitIntegerBooleans);
  if (!FoundOrErr)
    return FoundOrErr.takeError();
  if (!*FoundOrErr)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no AMDGPU metadata note");
  return std::move(Doc);
}

namespace detail {

// Map an `EF_AMDGPU_MACH_*` value to its canonical processor name
// using the X-macro list maintained in `llvm/BinaryFormat/ELF.h`. The
// table is the canonical source of truth — no per-processor table to
// keep in sync here.
inline llvm::StringRef machToProcessorName(unsigned Mach) {
  switch (Mach) {
#define X(NUM, ENUM, NAME)                                                     \
  case NUM:                                                                    \
    return NAME;
    AMDGPU_MACH_LIST(X)
#undef X
  default:
    return {};
  }
}

} // namespace detail

// Build the canonical AMDGPU ISA name from the ELF header alone (e.g.
// `amdgcn-amd-amdhsa--gfx1250:sramecc+:xnack-`). Supports all 4 ELF
// endian/bit combinations.
template <class ELFT>
inline llvm::Expected<std::string>
getElfIsaNameFromHeader(const llvm::object::ELFObjectFile<ELFT> &Obj) {
  using namespace llvm::ELF;
  auto Header = Obj.getELFFile().getHeader();

  if (Header.e_ident[EI_CLASS] != ELFCLASS64)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "ELFCLASS32 not supported by AMDGPU");
  if (Header.e_machine != EM_AMDGPU)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "ELF e_machine is not EM_AMDGPU");
  if (Header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "ELF OSABI is not AMDGPU_HSA");

  llvm::StringRef Processor =
      detail::machToProcessorName(Header.e_flags & EF_AMDGPU_MACH);
  if (Processor.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unrecognised AMDGPU MACH value");

  std::string IsaName = "amdgcn-amd-amdhsa--";
  IsaName += Processor.str();

  switch (Header.e_ident[EI_ABIVERSION]) {
  case ELFABIVERSION_AMDGPU_HSA_V4:
  case ELFABIVERSION_AMDGPU_HSA_V5:
  case ELFABIVERSION_AMDGPU_HSA_V6:
    switch (Header.e_flags & EF_AMDGPU_FEATURE_SRAMECC_V4) {
    case EF_AMDGPU_FEATURE_SRAMECC_OFF_V4:
      IsaName += ":sramecc-";
      break;
    case EF_AMDGPU_FEATURE_SRAMECC_ON_V4:
      IsaName += ":sramecc+";
      break;
    }
    switch (Header.e_flags & EF_AMDGPU_FEATURE_XNACK_V4) {
    case EF_AMDGPU_FEATURE_XNACK_OFF_V4:
      IsaName += ":xnack-";
      break;
    case EF_AMDGPU_FEATURE_XNACK_ON_V4:
      IsaName += ":xnack+";
      break;
    }
    break;
  default:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported AMDGPU ABI version");
  }
  return IsaName;
}

inline llvm::Expected<std::string>
getElfIsaNameFromHeader(llvm::MemoryBufferRef MB) {
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(MB);
  if (!ObjOrErr)
    return ObjOrErr.takeError();
  auto *Base = ObjOrErr->get();
  if (auto *E = llvm::dyn_cast<llvm::object::ELF32LEObjectFile>(Base))
    return getElfIsaNameFromHeader(*E);
  if (auto *E = llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(Base))
    return getElfIsaNameFromHeader(*E);
  if (auto *E = llvm::dyn_cast<llvm::object::ELF32BEObjectFile>(Base))
    return getElfIsaNameFromHeader(*E);
  if (auto *E = llvm::dyn_cast<llvm::object::ELF64BEObjectFile>(Base))
    return getElfIsaNameFromHeader(*E);
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported ELF variant");
}

} // namespace COMGR::co_utils

#endif // COMGR_CO_UTILS_H
