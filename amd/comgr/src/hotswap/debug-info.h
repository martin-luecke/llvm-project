//===- debug-info.h - DWARF preservation for hotswap raiser -------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Preserves DWARF debug info from a transpiled AMDGPU HSACO into the
// regenerated LLVM IR module. `KernelDwarfSource` loads the input and owns
// the parsed Binary + DWARFContext; `DebugInfoBuilder` mirrors its CU /
// subprogram metadata into the raised module and synthesises DILocations.
// Both are null / no-ops on input without `.debug_*` sections, so callers
// can instantiate them unconditionally.
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_DEBUG_INFO_H
#define HOTSWAP_TRANSPILER_DEBUG_INFO_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Support/MemoryBufferRef.h"

#include <memory>
#include <optional>
#include <string>

namespace llvm {
class DIBuilder;
class DICompileUnit;
class DIFile;
class DILocation;
class DISubprogram;
class DWARFContext;
class DWARFDie;
class Function;
class Module;
namespace object {
class Binary;
} // namespace object
} // namespace llvm

namespace COMGR::hotswap {

/// Loaded DWARF view of an input AMDGPU HSACO. Lookups take kernel-local
/// PC offsets (the raiser has no absolute VAs on hand) and key off the
/// kernel symbol VA internally.
class KernelDwarfSource {
public:
  /// Returns nullptr if `CodeObject` is not an ELF object or has no
  /// `.debug_info` section.
  static std::unique_ptr<KernelDwarfSource>
  create(llvm::MemoryBufferRef CodeObject);

  ~KernelDwarfSource();

  KernelDwarfSource(const KernelDwarfSource &) = delete;
  KernelDwarfSource &operator=(const KernelDwarfSource &) = delete;

  /// DWARF line info for `LocalPC` bytes into `KernelName`, or nullopt if
  /// the symbol is absent or the line table has no row for that PC.
  std::optional<llvm::DILineInfo> lookupLine(llvm::StringRef KernelName,
                                             uint64_t LocalPC) const;

  /// The DWARF subprogram DIE for `KernelName`, or an invalid DIE if not
  /// found.
  llvm::DWARFDie findSubprogram(llvm::StringRef KernelName) const;

  /// The compile unit DIE containing the subprogram for `KernelName`, or
  /// an invalid DIE if not found.
  llvm::DWARFDie findCompileUnit(llvm::StringRef KernelName) const;

  llvm::DWARFContext &context() const { return *Ctx; }

private:
  KernelDwarfSource();

  /// Absolute VA of `KernelName`, cached after first lookup; nullopt if
  /// the symbol is missing.
  std::optional<uint64_t> kernelVA(llvm::StringRef KernelName) const;

  std::unique_ptr<llvm::object::Binary> OwnedBinary;
  std::unique_ptr<llvm::DWARFContext> Ctx;

  // Symbol-name -> VA cache, lazily filled by kernelVA(). Keys are
  // StringRefs into the owned strings in CachedKernelNames.
  mutable llvm::DenseMap<llvm::StringRef, uint64_t> KernelVACache;
  mutable std::vector<std::string> CachedKernelNames;
};

/// Per-module DI-metadata factory, one instance per raised kernel. Wraps
/// `llvm::DIBuilder` and resolves DILocation requests against a source.
class DebugInfoBuilder {
public:
  /// Mirror the input's CU + subprogram for `KernelName` into `M`. Returns
  /// nullptr if the source has no DWARF subprogram for the kernel.
  static std::unique_ptr<DebugInfoBuilder>
  create(llvm::Module &M, const KernelDwarfSource &Source,
         llvm::StringRef KernelName);

  ~DebugInfoBuilder();

  DebugInfoBuilder(const DebugInfoBuilder &) = delete;
  DebugInfoBuilder &operator=(const DebugInfoBuilder &) = delete;

  /// Attach the mirrored DISubprogram to `Kernel`. Call once after the
  /// kernel Function is created.
  void attachSubprogramTo(llvm::Function &Kernel);

  /// DILocation for `LocalPC` bytes into the kernel, or nullptr if the PC
  /// has no line-table entry.
  llvm::DILocation *locationFor(uint64_t LocalPC) const;

  /// Emit DILocalVariable shells, run DIBuilder::finalize(), and set the
  /// "Debug Info Version" / "Dwarf Version" module flags if absent.
  void finalize();

private:
  DebugInfoBuilder(llvm::Module &M, const KernelDwarfSource &Source,
                   std::string KernelName);

  llvm::Module &M;
  const KernelDwarfSource &Source;
  std::string KernelName;

  std::unique_ptr<llvm::DIBuilder> DIB;
  llvm::DICompileUnit *CU = nullptr;
  llvm::DIFile *File = nullptr;
  llvm::DISubprogram *SP = nullptr;
  unsigned DwarfVersion = 5;
};

} // namespace COMGR::hotswap

#endif // HOTSWAP_TRANSPILER_DEBUG_INFO_H
