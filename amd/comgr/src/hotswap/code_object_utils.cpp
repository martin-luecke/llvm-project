//===- code_object_utils.cpp - Hotswap transpiler -------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "code_object_utils.h"

#include "../comgr-metadata.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <memory>

namespace COMGR::hotswap {

namespace {

// Wrap a raw `ArrayRef<uint8_t>` as an LLVM `MemoryBufferRef` without
// copying. The hotswap pipeline always passes ELF bytes that outlive
// the call, so a non-owning view is sufficient.
inline llvm::MemoryBufferRef toBufferRef(llvm::ArrayRef<uint8_t> Bytes) {
  return {llvm::StringRef(reinterpret_cast<const char *>(Bytes.data()),
                          Bytes.size()),
          /*BufferIdentifier=*/""};
}

// Copy `<kernelName>.kd`'s 64 KD bytes from .rodata into `Out`. The KD
// symbol is *always* in the .rodata section for amdhsa code objects (the
// AMDGPU asm printer emits it there); we map the symbol's virtual
// address to its file-level byte offset within the section's contents
// and copy the canonical 64-byte structure. Any mismatch (missing
// symbol, wrong size, address not within .rodata) is reported and
// produces `false`.
//
// We deliberately key off the symbol rather than the MsgPack metadata:
// the MsgPack notes do not include kernarg_preload_length /
// preload_offset, and that information is essential for modelling the
// gfx1250 user-SGPR ABI in Phase 4 of the raiser.
bool readKernelDescriptorBytes(llvm::object::ObjectFile &Obj,
                               llvm::StringRef KernelName,
                               std::array<uint8_t, 64> &Out) {
  std::string KdSymName = (KernelName + ".kd").str();

  std::optional<llvm::object::SectionRef> RodataSec;
  for (const auto &Sec : Obj.sections()) {
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) {
      (void)llvm::toString(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr == ".rodata") {
      RodataSec = Sec;
      break;
    }
  }
  if (!RodataSec) {
    llvm::errs() << "transpiler: readKernelDescriptorBytes: no .rodata "
                    "section in code object\n";
    return false;
  }

  uint64_t RodataAddr = RodataSec->getAddress();
  uint64_t RodataSize = RodataSec->getSize();
  auto RodataContentsOrErr = RodataSec->getContents();
  if (!RodataContentsOrErr) {
    (void)llvm::toString(RodataContentsOrErr.takeError());
    return false;
  }
  auto RodataContents = *RodataContentsOrErr;

  for (const auto &Sym : Obj.symbols()) {
    auto NameOrErr = Sym.getName();
    if (!NameOrErr) {
      (void)llvm::toString(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != KdSymName)
      continue;

    auto AddrOrErr = Sym.getAddress();
    if (!AddrOrErr) {
      (void)llvm::toString(AddrOrErr.takeError());
      return false;
    }
    uint64_t SymAddr = *AddrOrErr;

    if (SymAddr < RodataAddr || SymAddr + 64 > RodataAddr + RodataSize) {
      llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                   << KdSymName << "' at 0x" << llvm::utohexstr(SymAddr)
                   << " is not contained within .rodata [0x"
                   << llvm::utohexstr(RodataAddr) << ", 0x"
                   << llvm::utohexstr(RodataAddr + RodataSize) << ")\n";
      return false;
    }

    uint64_t Off = SymAddr - RodataAddr;
    if (Off + 64 > RodataContents.size()) {
      llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '"
                   << KdSymName << "' offset 0x" << llvm::utohexstr(Off)
                   << " + 64 exceeds .rodata contents size 0x"
                   << llvm::utohexstr(RodataContents.size()) << "\n";
      return false;
    }

    std::memcpy(Out.data(),
                reinterpret_cast<const uint8_t *>(RodataContents.data()) + Off,
                64);
    return true;
  }

  llvm::errs() << "transpiler: readKernelDescriptorBytes: symbol '" << KdSymName
               << "' not found\n";
  return false;
}

// Parse the four KD register fields we care about into `meta`. Wraps
// readKernelDescriptorBytes so the call site stays compact and the byte-
// offset constants are co-located with their usage.
void populateKernelDescriptorFields(llvm::object::ObjectFile &Obj,
                                    KernelMeta &Meta) {
  std::array<uint8_t, 64> KdBytes;
  if (!readKernelDescriptorBytes(Obj, Meta.Name, KdBytes)) {
    Meta.HasKernelDescriptor = false;
    return;
  }

  using namespace llvm::amdhsa;
  using llvm::support::endian::read16le;
  using llvm::support::endian::read32le;
  Meta.PrivateSegmentFixedSize =
      read32le(KdBytes.data() + PRIVATE_SEGMENT_FIXED_SIZE_OFFSET);
  Meta.ComputePgmRsrc1 = read32le(KdBytes.data() + COMPUTE_PGM_RSRC1_OFFSET);
  Meta.ComputePgmRsrc2 = read32le(KdBytes.data() + COMPUTE_PGM_RSRC2_OFFSET);
  Meta.KernelCodeProperties =
      read16le(KdBytes.data() + KERNEL_CODE_PROPERTIES_OFFSET);
  Meta.KernargPreload = read16le(KdBytes.data() + KERNARG_PRELOAD_OFFSET);
  Meta.HasKernelDescriptor = true;
}

// Look up `Key` in `Map`. Returns null when the key is absent.
// `MapDocNode::find(StringRef)` allocates the lookup key on `Map`'s
// owning document, so callers need only pass the literal string.
inline llvm::msgpack::DocNode *findInMap(llvm::msgpack::MapDocNode &Map,
                                         llvm::StringRef Key) {
  auto It = Map.find(Key);
  return (It == Map.end()) ? nullptr : &It->second;
}

// Pull a 64-bit integer value from a MsgPack node, accepting either
// signed or unsigned encoding (different toolchains emit either).
inline int64_t nodeAsInt(const llvm::msgpack::DocNode &N) {
  if (N.getKind() == llvm::msgpack::Type::Int)
    return N.getInt();
  if (N.getKind() == llvm::msgpack::Type::UInt)
    return static_cast<int64_t>(N.getUInt());
  return 0;
}

// Iterate the `amdhsa.kernels` array of a parsed AMDGPU MsgPack document
// and invoke `Fn` on each kernel map node. Stops on the first non-map
// child silently (matches the existing comgr metadata walker's tolerance).
template <class Fn>
void forEachKernelNode(llvm::msgpack::Document &Doc, Fn &&CB) {
  llvm::msgpack::DocNode &Root = Doc.getRoot();
  if (!Root.isMap())
    return;
  llvm::msgpack::DocNode *Kernels =
      findInMap(Root.getMap(), "amdhsa.kernels");
  if (!Kernels || !Kernels->isArray())
    return;
  for (auto &K : Kernels->getArray()) {
    if (!K.isMap())
      continue;
    CB(K.getMap());
  }
}

} // namespace

TextSection extractTextSection(llvm::ArrayRef<uint8_t> ElfData) {
  TextSection Result;
  llvm::MemoryBufferRef MB = toBufferRef(ElfData);
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(MB);
  if (!ObjOrErr) {
    llvm::errs() << "transpiler: Failed to parse ELF: "
                 << llvm::toString(ObjOrErr.takeError()) << "\n";
    return Result;
  }
  for (const auto &Sec : (*ObjOrErr)->sections()) {
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) {
      (void)llvm::toString(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != ".text")
      continue;
    auto ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      (void)llvm::toString(ContentsOrErr.takeError());
      continue;
    }
    Result.Bytes.assign(ContentsOrErr->begin(), ContentsOrErr->end());
    Result.Offset = Sec.getAddress();
    Result.Size = Sec.getSize();
    Result.Valid = true;
    return Result;
  }
  llvm::errs() << "transpiler: .text section not found in ELF\n";
  return Result;
}

std::vector<std::string> listKernelNames(llvm::ArrayRef<uint8_t> ElfData) {
  std::vector<std::string> Names;
  auto Doc = std::make_unique<llvm::msgpack::Document>();
  bool EmitIntegerBooleans = false;
  auto FoundOrErr = COMGR::metadata::walkElfMetadataIntoDoc(
      toBufferRef(ElfData), *Doc, EmitIntegerBooleans);
  if (!FoundOrErr) {
    llvm::errs() << "transpiler: listKernelNames: "
                 << llvm::toString(FoundOrErr.takeError()) << "\n";
    return Names;
  }
  if (!*FoundOrErr) {
    llvm::errs() << "transpiler: listKernelNames: no AMDGPU metadata note\n";
    return Names;
  }
  forEachKernelNode(*Doc, [&](llvm::msgpack::MapDocNode &KMap) {
    if (auto *N = findInMap(KMap, ".name"))
      Names.push_back(N->toString());
  });
  return Names;
}

KernelMeta extractKernelMeta(llvm::ArrayRef<uint8_t> ElfData,
                             llvm::StringRef KernelName) {
  KernelMeta Meta;

  llvm::MemoryBufferRef MB = toBufferRef(ElfData);
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(MB);
  if (!ObjOrErr) {
    llvm::errs() << "transpiler: extractKernelMeta: Failed to parse ELF: "
                 << llvm::toString(ObjOrErr.takeError()) << "\n";
    return Meta;
  }

  auto Doc = std::make_unique<llvm::msgpack::Document>();
  bool EmitIntegerBooleans = false;
  auto FoundOrErr =
      COMGR::metadata::walkElfMetadataIntoDoc(MB, *Doc, EmitIntegerBooleans);
  if (!FoundOrErr) {
    llvm::errs() << "transpiler: extractKernelMeta: "
                 << llvm::toString(FoundOrErr.takeError()) << "\n";
    return Meta;
  }
  if (!*FoundOrErr) {
    llvm::errs() << "transpiler: extractKernelMeta: no AMDGPU metadata note\n";
    return Meta;
  }
  bool MatchedKernel = false;
  forEachKernelNode(*Doc, [&](llvm::msgpack::MapDocNode &KMap) {
    if (MatchedKernel)
      return;
    auto *NameNode = findInMap(KMap, ".name");
    if (!NameNode || NameNode->toString() != KernelName)
      return;
    MatchedKernel = true;
    Meta.Name = NameNode->toString();

    if (auto *N = findInMap(KMap, ".kernarg_segment_size"))
      Meta.KernargSegmentSize = nodeAsInt(*N);
    if (auto *N = findInMap(KMap, ".group_segment_fixed_size"))
      Meta.GroupSegmentFixedSize = nodeAsInt(*N);
    if (auto *N = findInMap(KMap, ".private_segment_fixed_size"))
      Meta.PrivateSegmentFixedSize = nodeAsInt(*N);
    if (auto *N = findInMap(KMap, ".max_flat_workgroup_size"))
      Meta.MaxFlatWorkgroupSize = nodeAsInt(*N);

    if (auto *Args = findInMap(KMap, ".args");
        Args && Args->isArray()) {
      for (auto &ArgNode : Args->getArray()) {
        if (!ArgNode.isMap())
          continue;
        auto &AMap = ArgNode.getMap();
        KernelArgMeta Am;
        if (auto *N = findInMap(AMap, ".name"))
          Am.Name = N->toString();
        if (auto *N = findInMap(AMap, ".offset"))
          Am.Offset = nodeAsInt(*N);
        if (auto *N = findInMap(AMap, ".size"))
          Am.Size = nodeAsInt(*N);
        if (auto *N = findInMap(AMap, ".value_kind"))
          Am.ValueKind = N->toString();
        if (auto *N = findInMap(AMap, ".address_space"))
          Am.AddressSpace = nodeAsInt(*N);
        Meta.Args.push_back(Am);
      }
    }
  });

  if (!MatchedKernel) {
    llvm::errs() << "transpiler: extractKernelMeta: kernel '" << KernelName
                 << "' not found in metadata\n";
    return Meta;
  }

  // Fill the KD-register fields from .rodata. Sets Meta.HasKernelDescriptor
  // on success and emits a diagnostic on failure; the caller (raiser /
  // Phase-4 init) is responsible for refusing the lift if the field is
  // false rather than silently assuming a hardcoded SGPR layout.
  populateKernelDescriptorFields(*ObjOrErr->get(), Meta);
  return Meta;
}

llvm::Expected<uint64_t>
findKernelSymbolOffset(llvm::ArrayRef<uint8_t> ElfData,
                       llvm::StringRef KernelName) {
  llvm::MemoryBufferRef MB = toBufferRef(ElfData);
  auto ObjOrErr = llvm::object::ObjectFile::createELFObjectFile(MB);
  if (!ObjOrErr)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "findKernelSymbolOffset: Failed to parse ELF: " +
            llvm::toString(ObjOrErr.takeError()));

  uint64_t TextBase = UINT64_MAX;
  for (const auto &Sec : (*ObjOrErr)->sections()) {
    auto NameOrErr = Sec.getName();
    if (!NameOrErr)
      return NameOrErr.takeError();
    if (*NameOrErr == ".text") {
      TextBase = Sec.getAddress();
      break;
    }
  }
  if (TextBase == UINT64_MAX)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no .text section in ELF");

  for (const auto &Sym : (*ObjOrErr)->symbols()) {
    auto NameOrErr = Sym.getName();
    if (!NameOrErr) {
      (void)llvm::toString(NameOrErr.takeError());
      continue;
    }
    if (*NameOrErr != KernelName)
      continue;
    auto AddrOrErr = Sym.getAddress();
    if (!AddrOrErr)
      return AddrOrErr.takeError();
    if (*AddrOrErr < TextBase)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "symbol '" + KernelName + "' address < .text base");
    return *AddrOrErr - TextBase;
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "symbol '" + KernelName + "' not found in ELF");
}

std::string detectIsaFromElf(llvm::ArrayRef<uint8_t> ElfData) {
  std::string Isa;
  // Defer to amd_comgr's `getElfIsaNameFromBuffer`. Failure is silent here:
  // raise_cli calls this on the malformed-input path and falls back to a
  // filename heuristic when we return the empty string.
  if (COMGR::metadata::getElfIsaNameFromBuffer(toBufferRef(ElfData), Isa) !=
      AMD_COMGR_STATUS_SUCCESS) {
    return {};
  }
  return Isa;
}

} // namespace COMGR::hotswap
