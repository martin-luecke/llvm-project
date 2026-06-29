//===- topology_spoof.h - KFD topology gfx-version spoof ------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Builds patched copies of the KFD topology node `properties` files with the
/// `gfx_target_version` field overridden to a spoofed source target, and
/// exposes a real-path -> patched-path redirect map. The real device's gfx
/// version is recorded before the spoof so the transpiler downstream can target
/// it.
///
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_INTERPOSER_TOPOLOGY_SPOOF_H_
#define HOTSWAP_INTERPOSER_TOPOLOGY_SPOOF_H_

#include <cstdint>
#include <string>
#include <unordered_map>

namespace hotswap {
namespace interposer {

/// Parse a gfx target into the encoded `gfx_target_version` integer used by KFD
/// sysfs (major*10000 + minor*100 + stepping). Accepts either the encoded
/// integer ("125000") or a name ("gfx1250"). Returns 0 if unparseable.
uint32_t parseGfxTargetVersion(const char *Spec);

/// Render an encoded `gfx_target_version` as a "gfxNNNN" name (e.g. 125000 ->
/// "gfx1250"). Returns empty for 0.
std::string gfxTargetVersionName(uint32_t Version);

/// Surgical KFD topology spoof: redirect node `properties` reads to copies
/// whose `gfx_target_version` is overridden, leaving every other field at its
/// real value so the runtime still configures queues for the real hardware.
class TopologySpoof {
public:
  /// Build patched `properties` copies from the real KFD topology, overriding
  /// `gfx_target_version` to \p SpoofGfxVersion for every GPU node. Returns
  /// true if at least one GPU node was found and patched. On false the spoof is
  /// inert and `redirect` always returns empty.
  bool init(uint32_t SpoofGfxVersion);

  /// Return the patched path for \p RealPath, or empty if it is not redirected.
  std::string redirect(const char *RealPath) const;

  /// The real (un-spoofed) `gfx_target_version` of the first GPU node, or 0.
  uint32_t realGfxVersion() const { return RealGfxVersion; }

  /// True once at least one node `properties` file is being redirected.
  bool active() const { return !Redirects.empty(); }

  /// Remove the generated temp directory.
  void cleanup();

private:
  std::unordered_map<std::string, std::string> Redirects; // real -> patched
  uint32_t RealGfxVersion = 0;
  std::string TmpDir;
};

} // namespace interposer
} // namespace hotswap

#endif // HOTSWAP_INTERPOSER_TOPOLOGY_SPOOF_H_
