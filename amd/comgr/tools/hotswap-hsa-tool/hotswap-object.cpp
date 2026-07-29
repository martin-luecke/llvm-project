//===- hotswap-object.cpp - Source object eligibility --------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "hotswap-object.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace COMGR::hotswap::hsa_tool {
namespace {

struct AddressRange {
  uint64_t Begin = 0;
  uint64_t End = 0;
};

struct WritableStorageRanges {
  llvm::SmallVector<AddressRange, 4> Approved;
  llvm::SmallVector<AddressRange, 1> Dynamic;
  llvm::SmallVector<AddressRange, 1> RelroPadding;
};

bool makeAddressRange(uint64_t Address, uint64_t Size, AddressRange &Range) {
  if (Size == 0 || Address > std::numeric_limits<uint64_t>::max() - Size)
    return false;
  Range = {Address, Address + Size};
  return true;
}

bool contains(AddressRange Outer, AddressRange Inner) {
  return Inner.Begin >= Outer.Begin && Inner.End <= Outer.End;
}

bool rangesCover(AddressRange Required, llvm::ArrayRef<AddressRange> Ranges) {
  llvm::SmallVector<AddressRange, 4> InRange;
  for (AddressRange Range : Ranges)
    if (contains(Required, Range))
      InRange.push_back(Range);
  std::sort(InRange.begin(), InRange.end(),
            [](AddressRange Left, AddressRange Right) {
              return Left.Begin < Right.Begin;
            });
  uint64_t Covered = Required.Begin;
  for (AddressRange Range : InRange) {
    if (Range.End <= Covered)
      continue;
    if (Range.Begin > Covered)
      return false;
    Covered = Range.End;
  }
  return Covered == Required.End;
}

bool isKernelDescriptor(llvm::StringRef Name, uint64_t Size,
                        llvm::ArrayRef<std::string> Descriptors) {
  constexpr uint64_t KernelDescriptorSize = 64;
  return Size == KernelDescriptorSize &&
         llvm::any_of(Descriptors, [&](const std::string &Descriptor) {
           return Name == Descriptor;
         });
}

bool inspectStorageSymbols(const llvm::object::ObjectFile &Object,
                           llvm::ArrayRef<std::string> KernelDescriptors,
                           llvm::SmallVectorImpl<AddressRange> &HipMarkers,
                           std::string &Failure) {
  const auto InspectSymbol = [&](const llvm::object::SymbolRef &Symbol) {
    auto Flags = Symbol.getFlags();
    auto Name = Symbol.getName();
    if (!Flags) {
      Failure = "cannot inspect source ELF symbol flags: " +
                llvm::toString(Flags.takeError());
      return false;
    }
    if (!Name) {
      Failure = "cannot inspect a source ELF symbol name: " +
                llvm::toString(Name.takeError());
      return false;
    }
    if (*Flags & llvm::object::SymbolRef::SF_Undefined)
      return true;

    const llvm::object::ELFSymbolRef ElfSymbol(Symbol);
    const uint8_t Type = ElfSymbol.getELFType();
    if (Type != llvm::ELF::STT_OBJECT && Type != llvm::ELF::STT_COMMON)
      return true;
    const uint64_t Size = ElfSymbol.getSize();
    if (isKernelDescriptor(*Name, Size, KernelDescriptors)) {
      // The virtual executable currently presents agent-scoped program-linkage
      // kernels. Accept only the exact ELF linkage needed to synthesize that
      // contract without loading source machine code.
      if (ElfSymbol.getBinding() != llvm::ELF::STB_GLOBAL) {
        Failure = "source kernel descriptor '" + Name->str() +
                  "' does not have global linkage";
        return false;
      }
      return true;
    }
    if (Size == 1 && Name->starts_with("__hip_cuid_")) {
      auto Value = Symbol.getValue();
      if (!Value || *Value == std::numeric_limits<uint64_t>::max()) {
        Failure =
            "cannot inspect HIP compilation-unit marker '" + Name->str() + "'";
        if (!Value)
          llvm::consumeError(Value.takeError());
        return false;
      }
      const AddressRange Marker{*Value, *Value + 1};
      if (llvm::none_of(HipMarkers, [&](const AddressRange &Existing) {
            return Existing.Begin == Marker.Begin && Existing.End == Marker.End;
          }))
        HipMarkers.push_back(Marker);
      return true;
    }

    Failure = "source object uses unsupported device storage symbol '" +
              Name->str() + "'";
    return false;
  };

  for (const llvm::object::SymbolRef &Symbol : Object.symbols())
    if (!InspectSymbol(Symbol))
      return false;
  const auto *ElfObject =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(&Object);
  if (!ElfObject)
    return false;
  for (const llvm::object::ELFSymbolRef &DynamicSymbol :
       ElfObject->getDynamicSymbolIterators()) {
    const llvm::object::SymbolRef Symbol = DynamicSymbol;
    if (!InspectSymbol(Symbol))
      return false;
  }
  return true;
}

bool hipMarkersCoverSection(uint64_t Address, uint64_t Size,
                            llvm::ArrayRef<AddressRange> Markers) {
  AddressRange Section;
  if (!makeAddressRange(Address, Size, Section))
    return false;
  llvm::SmallVector<AddressRange, 2> InSection;
  for (const AddressRange &Marker : Markers)
    if (contains(Section, Marker))
      InSection.push_back(Marker);
  std::sort(InSection.begin(), InSection.end(),
            [](const AddressRange &Left, const AddressRange &Right) {
              return Left.Begin < Right.Begin;
            });
  uint64_t Covered = Address;
  for (const AddressRange &Marker : InSection) {
    if (Marker.Begin != Covered)
      return false;
    Covered = Marker.End;
  }
  return Covered == Section.End;
}

bool inspectWritableSections(const llvm::object::ObjectFile &Object,
                             llvm::ArrayRef<AddressRange> HipMarkers,
                             WritableStorageRanges &Ranges,
                             std::string &Failure) {
  for (const llvm::object::SectionRef &Section : Object.sections()) {
    const llvm::object::ELFSectionRef ElfSection(Section);
    const uint64_t Flags = ElfSection.getFlags();
    if ((Flags & (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE)) !=
        (llvm::ELF::SHF_ALLOC | llvm::ELF::SHF_WRITE))
      continue;
    auto Name = Section.getName();
    if (!Name) {
      Failure = "cannot inspect a writable source ELF section: " +
                llvm::toString(Name.takeError());
      return false;
    }
    AddressRange Range;
    if (!makeAddressRange(Section.getAddress(), Section.getSize(), Range)) {
      Failure = "writable source ELF section '" + Name->str() +
                "' has an invalid address range";
      return false;
    }
    if ((*Name == ".dynamic" &&
         ElfSection.getType() == llvm::ELF::SHT_DYNAMIC)) {
      Ranges.Approved.push_back(Range);
      Ranges.Dynamic.push_back(Range);
      continue;
    }
    if (*Name == ".relro_padding" &&
        ElfSection.getType() == llvm::ELF::SHT_NOBITS) {
      Ranges.Approved.push_back(Range);
      Ranges.RelroPadding.push_back(Range);
      continue;
    }
    if (*Name == ".bss" && ElfSection.getType() == llvm::ELF::SHT_NOBITS &&
        hipMarkersCoverSection(Section.getAddress(), Section.getSize(),
                               HipMarkers)) {
      Ranges.Approved.push_back(Range);
      continue;
    }

    Failure =
        "source object has unsupported writable section '" + Name->str() + "'";
    return false;
  }
  return true;
}

bool inspectWritableSegments(const llvm::object::ELF64LEObjectFile &Object,
                             const WritableStorageRanges &Ranges,
                             std::string &Failure) {
  auto Headers = Object.getELFFile().program_headers();
  if (!Headers) {
    Failure = "cannot inspect source ELF program headers: " +
              llvm::toString(Headers.takeError());
    return false;
  }

  llvm::SmallVector<AddressRange, 2> WritableLoads;
  llvm::SmallVector<AddressRange, 1> DynamicSegments;
  llvm::SmallVector<AddressRange, 1> RelroSegments;
  for (const auto &Header : *Headers) {
    if (Header.p_memsz == 0)
      continue;
    AddressRange Range;
    if (!makeAddressRange(Header.p_vaddr, Header.p_memsz, Range)) {
      Failure = "source ELF program header has an invalid address range";
      return false;
    }
    if (Header.p_type == llvm::ELF::PT_LOAD &&
        (Header.p_flags & llvm::ELF::PF_W)) {
      if (Header.p_filesz > Header.p_memsz ||
          (Header.p_flags & llvm::ELF::PF_X)) {
        Failure = "source ELF has an invalid writable load segment";
        return false;
      }
      WritableLoads.push_back(Range);
    } else if (Header.p_type == llvm::ELF::PT_DYNAMIC) {
      DynamicSegments.push_back(Range);
    } else if (Header.p_type == llvm::ELF::PT_GNU_RELRO) {
      RelroSegments.push_back(Range);
    }
  }

  for (AddressRange Load : WritableLoads) {
    if (!rangesCover(Load, Ranges.Approved)) {
      Failure = "source ELF writable load segment contains unsupported "
                "storage";
      return false;
    }
  }
  for (AddressRange Range : Ranges.Approved) {
    const bool InWritableLoad =
        llvm::any_of(WritableLoads,
                     [&](AddressRange Load) { return contains(Load, Range); });
    if (!InWritableLoad) {
      Failure = "source ELF writable section is outside a writable load "
                "segment";
      return false;
    }
  }
  for (AddressRange Range : Ranges.Dynamic) {
    if (llvm::none_of(DynamicSegments,
                      [&](AddressRange Segment) {
                        return Segment.Begin == Range.Begin &&
                               Segment.End == Range.End;
                      }) ||
        llvm::none_of(RelroSegments, [&](AddressRange Segment) {
          return contains(Segment, Range);
        })) {
      Failure = "source ELF .dynamic section is not loader-owned RELRO "
                "storage";
      return false;
    }
  }
  for (AddressRange Range : Ranges.RelroPadding) {
    if (llvm::none_of(RelroSegments, [&](AddressRange Segment) {
          return contains(Segment, Range);
        })) {
      Failure = "source ELF .relro_padding section is outside GNU_RELRO";
      return false;
    }
  }
  return true;
}

} // namespace

bool inspectSourceStorage(llvm::ArrayRef<uint8_t> Object,
                          llvm::ArrayRef<std::string> KernelDescriptors,
                          std::string &Failure) {
  const llvm::StringRef Bytes(reinterpret_cast<const char *>(Object.data()),
                              Object.size());
  auto ObjectOrError = llvm::object::ObjectFile::createObjectFile(
      llvm::MemoryBufferRef(Bytes, "hotswap-source-object"));
  if (!ObjectOrError) {
    Failure =
        "cannot parse source ELF: " + llvm::toString(ObjectOrError.takeError());
    return false;
  }
  const auto *Elf =
      llvm::dyn_cast<llvm::object::ELF64LEObjectFile>(ObjectOrError->get());
  if (!Elf || Elf->getELFFile().getHeader().e_machine != llvm::ELF::EM_AMDGPU ||
      Elf->getELFFile().getHeader().e_type != llvm::ELF::ET_DYN) {
    Failure = "source code object is not an ELF64 little-endian AMDGPU "
              "shared object";
    return false;
  }

  llvm::SmallVector<AddressRange, 2> HipMarkers;
  WritableStorageRanges Ranges;
  return inspectStorageSymbols(*Elf, KernelDescriptors, HipMarkers, Failure) &&
         inspectWritableSections(*Elf, HipMarkers, Ranges, Failure) &&
         inspectWritableSegments(*Elf, Ranges, Failure);
}

} // namespace COMGR::hotswap::hsa_tool
