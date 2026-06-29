// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer.cpp
/// @brief LD_PRELOAD interposer that presents a spoofed gfx target to the ROCm
/// stack while forwarding all real device traffic to the real driver.
///
/// @details Reads of the KFD topology `properties` file are redirected to a
/// copy whose `gfx_target_version` is overridden, so the whole stack selects
/// and emits the spoofed source ISA. The real /dev/kfd and /dev/dri are left
/// untouched (we forward to real hardware rather than emulate it), and every
/// other path falls through to libc.
///
/// Identifiers here intentionally follow the libc/POSIX idiom (lowercase symbol
/// and parameter names) rather than the surrounding Comgr CamelCase convention:
/// the interposed entry points must keep the exact libc symbol names, and
/// naming the function-pointer table and parameters after their libc
/// counterparts keeps the hooks readable against the man-page signatures they
/// shadow.

#include "topology_spoof.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

namespace {

template <typename Fn> Fn lookupNext(const char *Name) {
  return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, Name));
}

/// Real libc function pointers resolved via dlsym(RTLD_NEXT).
class LibcPassthrough {
public:
  int (*openat)(int, const char *, int, ...) = nullptr;
  FILE *(*fopen)(const char *, const char *) = nullptr;
  FILE *(*freopen)(const char *, const char *, FILE *) = nullptr;
  int (*stat)(const char *, struct stat *) = nullptr;
  int (*lstat)(const char *, struct stat *) = nullptr;
  int (*access)(const char *, int) = nullptr;

  void resolve() {
    openat = lookupNext<decltype(openat)>("openat");
    fopen = lookupNext<decltype(fopen)>("fopen");
    freopen = lookupNext<decltype(freopen)>("freopen");
    stat = lookupNext<decltype(stat)>("stat");
    lstat = lookupNext<decltype(lstat)>("lstat");
    access = lookupNext<decltype(access)>("access");
  }
};

// Function-local statics, not namespace globals: an LD_PRELOAD library's hooks
// can run before (and its globals' constructors after) the constructor that
// would populate them. Construct-on-first-use sidesteps that ordering hazard.
LibcPassthrough &real() {
  static LibcPassthrough R = [] {
    LibcPassthrough Tmp;
    Tmp.resolve();
    return Tmp;
  }();
  return R;
}

hotswap::interposer::TopologySpoof &spoof() {
  static hotswap::interposer::TopologySpoof S;
  return S;
}

bool logEnabled() {
  static bool Enabled = std::getenv("HOTSWAP_INTERPOSER_LOG") != nullptr ||
                        std::getenv("HSA_HOTSWAP_VERBOSE") != nullptr;
  return Enabled;
}

/// Return the spoofed redirect target for \p Path, or empty if it is not a
/// redirected topology file. Taking a plain (non-`__nonnull`) parameter keeps
/// the defensive null check from tripping glibc's nonnull-compare warnings.
std::string redirectOf(const char *Path) {
  if (!Path)
    return {};
  return spoof().redirect(Path);
}

int fopenFlagsFromMode(const char *Mode) {
  bool Plus = std::strchr(Mode, '+') != nullptr;
  switch (Mode[0]) {
  case 'w':
    return (Plus ? O_RDWR : O_WRONLY) | O_CREAT | O_TRUNC;
  case 'a':
    return (Plus ? O_RDWR : O_WRONLY) | O_CREAT | O_APPEND;
  default:
    return Plus ? O_RDWR : O_RDONLY;
  }
}

__attribute__((constructor)) void initInterposer() {
  real();

  const char *Spec = std::getenv("HOTSWAP_INTERPOSER_SPOOF");
  if (!Spec || !*Spec)
    return; // Inert: pure passthrough.

  uint32_t SpoofVersion = hotswap::interposer::parseGfxTargetVersion(Spec);
  if (SpoofVersion == 0) {
    if (logEnabled())
      std::fprintf(stderr,
                   "[hotswap-interposer] invalid HOTSWAP_INTERPOSER_SPOOF=%s\n",
                   Spec);
    return;
  }

  if (!spoof().init(SpoofVersion)) {
    if (logEnabled())
      std::fprintf(
          stderr,
          "[hotswap-interposer] no GPU topology node to spoof; inert\n");
    return;
  }

  // Publish the real device beneath the spoof as the HotSwap transpile target.
  // Both the in-process tool half and a future native-runtime HotSwap read
  // HSA_HOTSWAP_TARGET, so the spoof and the transpiler agree on the real ISA
  // even though the agent now reports the spoofed source. Do not clobber a
  // value the user set explicitly.
  std::string RealName =
      hotswap::interposer::gfxTargetVersionName(spoof().realGfxVersion());
  if (!RealName.empty())
    setenv("HSA_HOTSWAP_TARGET", RealName.c_str(), /*overwrite=*/0);

  if (logEnabled()) {
    std::string SourceName =
        hotswap::interposer::gfxTargetVersionName(SpoofVersion);
    const char *Target = std::getenv("HSA_HOTSWAP_TARGET");
    std::fprintf(stderr,
                 "[hotswap-interposer] spoofing %s (real device %s); "
                 "HSA_HOTSWAP_TARGET=%s\n",
                 SourceName.c_str(), RealName.c_str(), Target ? Target : "");
  }
}

} // namespace

extern "C" {

int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  std::string redirected = redirectOf(path);
  if (!redirected.empty())
    return real().openat(AT_FDCWD, redirected.c_str(), flags, mode);
  return real().openat(AT_FDCWD, path, flags, mode);
}

int open64(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return open(path, flags, mode);
}

int __open_2(const char *path, int oflag) { return open(path, oflag, 0); }
int __open64_2(const char *path, int oflag) { return open(path, oflag, 0); }

int openat(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  std::string redirected = redirectOf(path);
  if (!redirected.empty())
    return real().openat(AT_FDCWD, redirected.c_str(), flags, mode);
  return real().openat(dirfd, path, flags, mode);
}

int openat64(int dirfd, const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = static_cast<mode_t>(va_arg(ap, int));
    va_end(ap);
  }
  return openat(dirfd, path, flags, mode);
}

int __openat_2(int dirfd, const char *path, int oflag) {
  return openat(dirfd, path, oflag, 0);
}
int __openat64_2(int dirfd, const char *path, int oflag) {
  return openat(dirfd, path, oflag, 0);
}

FILE *fopen(const char *path, const char *mode) {
  std::string redirected = redirectOf(path);
  if (!redirected.empty()) {
    int fd = real().openat(AT_FDCWD, redirected.c_str(),
                           fopenFlagsFromMode(mode), 0644);
    if (fd < 0)
      return nullptr;
    return fdopen(fd, mode);
  }
  return real().fopen(path, mode);
}

FILE *fopen64(const char *path, const char *mode) { return fopen(path, mode); }

FILE *freopen(const char *path, const char *mode, FILE *stream) {
  std::string redirected = redirectOf(path);
  if (!redirected.empty())
    return real().freopen(redirected.c_str(), mode, stream);
  return real().freopen(path, mode, stream);
}

FILE *freopen64(const char *path, const char *mode, FILE *stream) {
  return freopen(path, mode, stream);
}

int stat(const char *path, struct stat *buf) {
  std::string redirected = redirectOf(path);
  if (!redirected.empty())
    return real().stat(redirected.c_str(), buf);
  return real().stat(path, buf);
}

int lstat(const char *path, struct stat *buf) {
  std::string redirected = redirectOf(path);
  if (!redirected.empty())
    return real().lstat(redirected.c_str(), buf);
  return real().lstat(path, buf);
}

int access(const char *path, int mode) {
  std::string redirected = redirectOf(path);
  if (!redirected.empty())
    return real().access(redirected.c_str(), mode);
  return real().access(path, mode);
}

// glibc exposes the *64 and __?xstat ABI variants separately; libraries built
// against different glibc versions may import any of them, so each is
// redirected the same way through a locally resolved real entry point.
int stat64(const char *path, struct stat64 *buf) {
  using Fn = int (*)(const char *, struct stat64 *);
  static Fn RealStat64 = lookupNext<Fn>("stat64");
  if (!RealStat64)
    return -1;
  std::string redirected = redirectOf(path);
  return RealStat64(redirected.empty() ? path : redirected.c_str(), buf);
}

int lstat64(const char *path, struct stat64 *buf) {
  using Fn = int (*)(const char *, struct stat64 *);
  static Fn RealLstat64 = lookupNext<Fn>("lstat64");
  if (!RealLstat64)
    return -1;
  std::string redirected = redirectOf(path);
  return RealLstat64(redirected.empty() ? path : redirected.c_str(), buf);
}

int __xstat(int ver, const char *path, struct stat *buf) {
  using Fn = int (*)(int, const char *, struct stat *);
  static Fn RealXstat = lookupNext<Fn>("__xstat");
  if (!RealXstat)
    return -1;
  std::string redirected = redirectOf(path);
  return RealXstat(ver, redirected.empty() ? path : redirected.c_str(), buf);
}

int __xstat64(int ver, const char *path, struct stat64 *buf) {
  using Fn = int (*)(int, const char *, struct stat64 *);
  static Fn RealXstat64 = lookupNext<Fn>("__xstat64");
  if (!RealXstat64)
    return -1;
  std::string redirected = redirectOf(path);
  return RealXstat64(ver, redirected.empty() ? path : redirected.c_str(), buf);
}

int __lxstat(int ver, const char *path, struct stat *buf) {
  using Fn = int (*)(int, const char *, struct stat *);
  static Fn RealLxstat = lookupNext<Fn>("__lxstat");
  if (!RealLxstat)
    return -1;
  std::string redirected = redirectOf(path);
  return RealLxstat(ver, redirected.empty() ? path : redirected.c_str(), buf);
}

int __lxstat64(int ver, const char *path, struct stat64 *buf) {
  using Fn = int (*)(int, const char *, struct stat64 *);
  static Fn RealLxstat64 = lookupNext<Fn>("__lxstat64");
  if (!RealLxstat64)
    return -1;
  std::string redirected = redirectOf(path);
  return RealLxstat64(ver, redirected.empty() ? path : redirected.c_str(), buf);
}

} // extern "C"
