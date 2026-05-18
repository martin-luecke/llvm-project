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

bool getOCMLDeviceLibraryNames(llvm::StringRef TargetIsa,
                               unsigned TargetWaveSize,
                               llvm::SmallVectorImpl<std::string> &Names,
                               std::string &Error) {
  Names.clear();

  if (!TargetIsa.consume_front("gfx")) {
    Error = (Twine("target ISA '") + TargetIsa +
             "' does not name a gfx processor").str();
    return false;
  }

  if (TargetWaveSize != 32 && TargetWaveSize != 64) {
    Error = (Twine("cannot select OCML wavefront-size control library for "
                   "target wave size ") +
             Twine(TargetWaveSize)).str();
    return false;
  }

  std::string IsaSuffix = TargetIsa.str();
  for (char &C : IsaSuffix) {
    if (C == '-')
      C = '_';
  }

  Names.push_back("ocml.bc");
  Names.push_back("ockl.bc");
  Names.push_back("oclc_abi_version_600.bc");
  Names.push_back("oclc_isa_version_" + IsaSuffix + ".bc");
  Names.push_back("oclc_finite_only_off.bc");
  Names.push_back("oclc_unsafe_math_off.bc");
  Names.push_back(TargetWaveSize == 64 ? "oclc_wavefrontsize64_on.bc"
                                       : "oclc_wavefrontsize64_off.bc");
  return true;
}

} // namespace COMGR
