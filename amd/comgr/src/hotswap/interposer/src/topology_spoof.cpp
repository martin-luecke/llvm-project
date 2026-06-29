//===- topology_spoof.cpp - KFD topology gfx-version spoof ----------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "topology_spoof.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace hotswap {
namespace interposer {

namespace {

constexpr const char *KfdTopologyRoot = "/sys/devices/virtual/kfd/kfd/topology";
constexpr const char *KfdTopologyAlias = "/sys/class/kfd/kfd/topology";
constexpr std::string_view GfxField = "gfx_target_version";

/// Read \p Path into a string. Returns false on any error. While the spoof map
/// is still empty (construction time) the interposer's own hooks are inert, so
/// these reads reach the real sysfs files.
bool readFile(const std::string &Path, std::string &Out) {
  std::ifstream In(Path, std::ios::binary);
  if (!In)
    return false;
  std::ostringstream Ss;
  Ss << In.rdbuf();
  Out = Ss.str();
  return In.good() || In.eof();
}

bool writeFile(const std::string &Path, const std::string &Content) {
  std::ofstream Os(Path, std::ios::binary | std::ios::trunc);
  if (!Os)
    return false;
  Os << Content;
  return Os.good();
}

/// Find the value of the `gfx_target_version` line in a properties blob.
/// Returns 0 if the field is absent or zero.
uint32_t parseGfxField(const std::string &Props) {
  std::istringstream Ss(Props);
  std::string Name;
  while (Ss >> Name) {
    if (Name == GfxField) {
      unsigned long long Val = 0;
      Ss >> Val;
      return static_cast<uint32_t>(Val);
    }
    Ss.ignore(0x7fffffff, '\n');
  }
  return 0;
}

/// Replace the `gfx_target_version` value in \p Props with \p NewVersion,
/// preserving the rest of the file byte-for-byte.
std::string patchGfxField(const std::string &Props, uint32_t NewVersion) {
  std::string Out;
  Out.reserve(Props.size() + 8);
  size_t Pos = 0;
  while (Pos < Props.size()) {
    size_t Eol = Props.find('\n', Pos);
    if (Eol == std::string::npos)
      Eol = Props.size();
    std::string_view Line(Props.data() + Pos, Eol - Pos);
    if (Line.substr(0, GfxField.size()) == GfxField &&
        (Line.size() == GfxField.size() || Line[GfxField.size()] == ' ')) {
      Out += GfxField;
      Out += ' ';
      Out += std::to_string(NewVersion);
    } else {
      Out += Line;
    }
    if (Eol < Props.size())
      Out += '\n';
    Pos = Eol + 1;
  }
  return Out;
}

bool makeDir(const std::string &Path) {
  return ::mkdir(Path.c_str(), 0755) == 0 || errno == EEXIST;
}

} // namespace

uint32_t parseGfxTargetVersion(const char *Spec) {
  if (!Spec || !*Spec)
    return 0;
  std::string_view Sv(Spec);
  if (Sv.substr(0, 3) == "gfx")
    Sv.remove_prefix(3);
  for (char C : Sv)
    if (!std::isdigit(static_cast<unsigned char>(C)))
      return 0;
  if (Sv.empty())
    return 0;
  unsigned long long Num = std::strtoull(std::string(Sv).c_str(), nullptr, 10);
  // A plain encoded version (major*10000+minor*100+stepping) is >= 10000 for
  // any real AMD gfx target; a short "gfxNNN(N)" name is the digit string
  // itself where the last digit is the stepping, the next the minor, the rest
  // major.
  if (Num >= 10000)
    return static_cast<uint32_t>(Num);
  if (Sv.size() < 3)
    return 0;
  uint32_t Stepping = Sv[Sv.size() - 1] - '0';
  uint32_t Minor = Sv[Sv.size() - 2] - '0';
  uint32_t Major = static_cast<uint32_t>(std::strtoul(
      std::string(Sv.substr(0, Sv.size() - 2)).c_str(), nullptr, 10));
  return Major * 10000 + Minor * 100 + Stepping;
}

std::string gfxTargetVersionName(uint32_t Version) {
  if (Version == 0)
    return {};
  uint32_t Major = Version / 10000;
  uint32_t Minor = (Version / 100) % 100;
  uint32_t Stepping = Version % 100;
  std::string Name = "gfx" + std::to_string(Major);
  // Single-digit minor/stepping are rendered without separators, matching the
  // canonical gfxNNNN naming for the supported targets.
  Name += std::to_string(Minor);
  Name += std::to_string(Stepping);
  return Name;
}

bool TopologySpoof::init(uint32_t SpoofGfxVersion) {
  if (SpoofGfxVersion == 0)
    return false;

  std::string NodesRoot = std::string(KfdTopologyRoot) + "/nodes";
  DIR *Dir = ::opendir(NodesRoot.c_str());
  if (!Dir)
    return false;

  std::vector<std::string> NodeIds;
  while (dirent *Ent = ::readdir(Dir)) {
    std::string_view Name(Ent->d_name);
    if (Name.empty() || !std::all_of(Name.begin(), Name.end(), [](char C) {
          return std::isdigit((unsigned char)C);
        }))
      continue;
    NodeIds.emplace_back(Name);
  }
  ::closedir(Dir);

  // Stage the patched copies under a private temp directory.
  char Tmpl[] = "/tmp/hotswap_topo_XXXXXX";
  char *Made = ::mkdtemp(Tmpl);
  if (!Made)
    return false;
  std::string Tmp(Made);

  std::unordered_map<std::string, std::string> Pending;
  uint32_t RealVersion = 0;
  for (const std::string &Id : NodeIds) {
    std::string RealProps = NodesRoot + "/" + Id + "/properties";
    std::string Props;
    if (!readFile(RealProps, Props))
      continue;
    uint32_t Gfx = parseGfxField(Props);
    if (Gfx == 0)
      continue; // CPU node or no GPU gfx version.
    if (RealVersion == 0)
      RealVersion = Gfx;

    std::string NodeDir = Tmp + "/nodes/" + Id;
    if (!makeDir(Tmp + "/nodes") || !makeDir(NodeDir))
      continue;
    std::string PatchedProps = NodeDir + "/properties";
    if (!writeFile(PatchedProps, patchGfxField(Props, SpoofGfxVersion)))
      continue;
    Pending.emplace(RealProps, PatchedProps);
  }

  if (Pending.empty()) {
    ::rmdir(Tmp.c_str());
    return false;
  }

  TmpDir = Tmp;
  RealGfxVersion = RealVersion;
  Redirects = std::move(Pending);
  return true;
}

std::string TopologySpoof::redirect(const char *RealPath) const {
  if (!RealPath || Redirects.empty())
    return {};
  std::string_view Sv(RealPath);
  // Normalize the /sys/class/kfd alias to the canonical topology root.
  std::string Canonical;
  if (Sv.substr(0, std::strlen(KfdTopologyAlias)) == KfdTopologyAlias) {
    Canonical = std::string(KfdTopologyRoot) +
                std::string(Sv.substr(std::strlen(KfdTopologyAlias)));
    RealPath = Canonical.c_str();
  }
  auto It = Redirects.find(RealPath);
  return It != Redirects.end() ? It->second : std::string{};
}

void TopologySpoof::cleanup() {
  for (const auto &Kv : Redirects)
    ::unlink(Kv.second.c_str());
  Redirects.clear();
  // Best-effort directory teardown; nodes/<id> dirs then the roots.
  if (!TmpDir.empty()) {
    // Leave residue rather than risk recursive removal of an unexpected path.
    std::string Nodes = TmpDir + "/nodes";
    DIR *D = ::opendir(Nodes.c_str());
    if (D) {
      while (dirent *E = ::readdir(D)) {
        std::string_view N(E->d_name);
        if (N == "." || N == "..")
          continue;
        ::rmdir((Nodes + "/" + std::string(N)).c_str());
      }
      ::closedir(D);
    }
    ::rmdir(Nodes.c_str());
    ::rmdir(TmpDir.c_str());
  }
}

} // namespace interposer
} // namespace hotswap
