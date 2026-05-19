//===- comgr-device-libs.cpp - Handle AMD Device Libraries ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the handling of the AMD Device Libraries, which are
/// LLVM IR objects embedded into Comgr via header files.
///
/// We also handle OpenCL pre-compiled headers, which are similarly embedded in
/// Comgr.
///
//===----------------------------------------------------------------------===//

#include "comgr-device-libs.h"
#include "comgr.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <cstdint>
#include <optional>

using namespace llvm;

namespace COMGR {

namespace {
#include "libraries.inc"
#include "libraries_sha.inc"
#include "opencl-c-base.inc"
} // namespace

ArrayRef<unsigned char> getDeviceLibrariesIdentifier() {
  return DEVICE_LIBS_ID;
}

StringRef getOpenCLCBaseHeaderContents() {
  return StringRef(reinterpret_cast<const char *>(opencl_c_base),
                   opencl_c_base_size);
}

llvm::ArrayRef<std::tuple<llvm::StringRef, llvm::StringRef>>
getDeviceLibraries() {
  static std::tuple<llvm::StringRef, llvm::StringRef> DeviceLibs[] = {
#define AMD_DEVICE_LIBS_TARGET(target)                                         \
  {#target ".bc",                                                              \
   llvm::StringRef(reinterpret_cast<const char *>(target##_lib),               \
                   target##_lib_size)},
#include "libraries_defs.inc"
  };
  return DeviceLibs;
}

namespace {

struct IsaLibraryEntry {
  llvm::StringRef GfxIp;
  llvm::StringRef Name;
};

llvm::ArrayRef<IsaLibraryEntry> getIsaLibraryEntries() {
  static const IsaLibraryEntry Entries[] = {
#define AMD_DEVICE_LIBS_TARGET(target)
#define AMD_DEVICE_LIBS_GFXIP(target, gfxip)                                   \
  {gfxip, #target ".bc"},
#define AMD_DEVICE_LIBS_FUNCTION(target, function)
#include "libraries_defs.inc"
  };
  return Entries;
}

std::optional<llvm::StringRef> selectIsaLibraryName(llvm::StringRef GfxIp) {
  for (const IsaLibraryEntry &Entry : getIsaLibraryEntries()) {
    if (Entry.GfxIp == GfxIp)
      return Entry.Name;
  }
  return std::nullopt;
}

bool hasEmbeddedDeviceLibrary(llvm::StringRef Name) {
  for (const auto &Lib : getDeviceLibraries()) {
    if (std::get<0>(Lib) == Name)
      return true;
  }
  return false;
}

bool validateSelectedDeviceLibraries(llvm::ArrayRef<llvm::StringRef> Names,
                                     std::string &Error) {
  for (llvm::StringRef Name : Names) {
    if (hasEmbeddedDeviceLibrary(Name))
      continue;
    Error = (Twine("selected OCML device library '") + Name +
             "' is not embedded in this COMGR build")
                .str();
    return false;
  }
  return true;
}

} // namespace

bool getOCMLDeviceLibraryNames(llvm::StringRef TargetProcessor,
                               unsigned TargetWaveSize,
                               llvm::SmallVectorImpl<std::string> &Names,
                               std::string &Error) {
  Names.clear();

  llvm::StringRef GfxIp = TargetProcessor;
  if (!GfxIp.consume_front("gfx") || GfxIp.empty()) {
    Error = (Twine("target processor '") + TargetProcessor +
             "' does not name a gfx processor").str();
    return false;
  }

  std::optional<llvm::StringRef> IsaLibraryName =
      selectIsaLibraryName(GfxIp);
  if (!IsaLibraryName) {
    Error = (Twine("no embedded OCML ISA control library for target processor '") +
             TargetProcessor + "'")
                .str();
    return false;
  }

  if (TargetWaveSize != 32 && TargetWaveSize != 64) {
    Error = (Twine("cannot select OCML wavefront-size control library for "
                   "target wave size ") +
             Twine(TargetWaveSize)).str();
    return false;
  }

  llvm::SmallVector<llvm::StringRef, 8> Selected = {
      "ocml.bc",
      "ockl.bc",
      "oclc_abi_version_600.bc",
      *IsaLibraryName,
      "oclc_finite_only_off.bc",
      "oclc_unsafe_math_off.bc",
      TargetWaveSize == 64 ? "oclc_wavefrontsize64_on.bc"
                           : "oclc_wavefrontsize64_off.bc",
  };
  if (!validateSelectedDeviceLibraries(Selected, Error))
    return false;

  for (llvm::StringRef Name : Selected)
    Names.push_back(Name.str());
  return true;
}

} // namespace COMGR
