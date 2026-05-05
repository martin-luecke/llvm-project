//===- raise_cli.cpp - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Per-file raiser CLI — two modes.
//
// Usage:
//   raise_cli <code-object.co|.hsaco> [--isa=<arch>] [--target-isa=<arch>]
//   raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] [--isa=<arch>]
//                                     [--target-isa=<arch>]
//
// Default (kerneldex-coverage) mode. For each kernel in the code object,
// forks a child that runs raiseToIR so that a fatal error
// (report_fatal_error / asan trap / ...) in one kernel doesn't poison
// the whole file.  Emits one line per kernel on stdout:
//
//   OK   <kernel-name> (<lifted>/<total>)
//   FAIL <kernel-name> -> <mnemonic> [<format>]
//
// Kernels that crashed in the child (signal, non-zero exit, incomplete
// shm) are reported as a FAIL with mnemonic ``__crash__`` and format
// ``signal_<N>`` so they still land in the kerneldex worklist instead of
// being silently dropped.
//
// Exits 0 iff every kernel succeeded; otherwise 1.  ISA is auto-detected
// from the filename (look for ``gfx<digits>[a-z]?``) when ``--isa=`` is
// not passed.
//
// --emit-ir mode. Designed for lit tests. Runs raiseToIR in-process (no
// fork), dumps the raised LLVM IR for a single kernel on stdout, and
// leaves stderr alone so FileCheck can match warnings / abort-gate
// diagnostics. Selects the only kernel when the code object has one, or
// requires the ``=<kernel>`` form when there are multiple. Exits 0 iff
// the kernel raised successfully; non-zero otherwise.
//
// --target-isa=<arch>. Optional. Controls the target ISA the raiser
// lowers for; defaults to the source ISA (same-wave translation). Use
// to exercise cross-wave paths from a single CO (e.g. a gfx1250 CO
// compiled for a wave64 target).
//
// --enable-writelane-rewrite / --disable-writelane-rewrite. Default
// **on** (post-Triton-corpus graduation; see raiser.h for the full
// rationale).  Controls the post-raise rewrite of cross-widen-divergent
// `v_writelane_b32` / `v_readlane_b32` sites into per-source-wave
// `select` / `ds_bpermute` primitives — see
// `rewrite_cross_lane_divergent.{hpp,cpp}` and
// hotswap/docs/wave-size-translation.md §5.6.3.
//
// `--enable-writelane-rewrite` is accepted for backward compatibility
// (the canonical flag name used by existing lit fixtures) and is a
// no-op since the default is already on; `--disable-writelane-rewrite`
// forces the pre-rewrite path and is used by the `REFUSE` / `UNCHANGED`
// sibling RUN lines in the writelane/readlane regression fixtures to
// pin the pre-rewrite contract.  Later-wins between the two flags is
// by command-line order (last occurrence decides).
//
// --enable-wave-native / --disable-wave-native. Default **on** as
// of the WaveNative graduation. Selects `WaveNativeProjection`
// instead of `ModuloReplicationProjection` for wave32 source →
// wave64 target cross-widening. Under wave-native the kernel entry
// emits `@llvm.amdgcn.init_whole_wave` so hardware EXEC = -1 for
// the body, which:
//   * makes the WMMA → MFMA pipeline in `wmma_lowering.cpp`
//     correct on the upper half of the Wave64 target (the original
//     design motivation — see wave-size-translation.md §5.6.1);
//   * projects kernels with `num_warps > 1` correctly by giving
//     each target lane its own modeled-EXEC bit (fixes the
//     `swiglu_fp32` / `corpus_layernorm_fp32` class documented in
//     hotswap/docs/modrep-predicate-chain.md §4.3 sub-case 1);
//   * renders the C5 classifier's MODREP-specific refusal
//     rationale inapplicable — target lanes have their own
//     modeled-EXEC bits rather than sharing source wave 0's. The
//     classifier's `waveNative` gate suppresses refusal on this
//     path. For `canary_bpermute_scan_fp32`, the underlying
//     miscompile that would otherwise surface is closed by the
//     VOPD-cndmask SGPR-condition fix
//     (modrep-predicate-chain.md §6.4) rather than by the
//     projection choice itself.
//
// `--disable-wave-native` opts back into `ModuloReplicationProjection`
// for the narrow class of pointwise / independent-half kernels where
// MODREP's "replicas of source wave 0" model is correct AND where
// the C5 refusal under MODREP is the desired loud-fail signal.
// No env-var override exists; `HSA_HOTSWAP_WAVE_NATIVE` was a
// transient test hook during the graduation sweep and has been
// removed so the opt-out path isn't silently bypassed.
//
// (The earlier `--enable-permlane16-xor3-partner` /
// `--enable-permlane16-swap-selfpreserve` flags were removed along
// with their rewrite passes once the asymmetric
// `v_permlane16_swap_b32` lift landed — see
// `handle_valu_cross_lane.cpp::emitPermLaneSwapEmulation` and
// matrix-translation.md §12.4.7.)

#include "hotswap/code_object_utils.h"
#include "hotswap/pipeline.h"
#include "hotswap/raiser.h"

// raiser.h forward-declares llvm::LLVMContext and llvm::Module but
// RaiseResult holds them by unique_ptr, so the destructor synthesized in
// main() needs the complete types.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Per-kernel crash isolation in the all-kernels coverage path uses fork() +
// MAP_ANONYMOUS shared memory + waitpid() — none of which exist on Windows.
// `--emit-ir=<NAME>` (the lit-fixture path) raises a single kernel
// in-process and is unaffected. The Windows fallback in the all-kernels
// branch raises in-process too: a crash takes the whole tool down rather
// than turning into a FAIL row, but the build works and per-kernel OK/FAIL
// reporting is preserved.
#if defined(LLVM_ON_UNIX)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

// Shared-memory block handed from each per-kernel child back to the parent.
// Using a fixed-size POD struct keeps the IPC trivially safe across fork().
struct KernelRaiseStats {
  bool Done;
  bool Success;
  int Lifted;
  int Total;
  char FailMnemonic[128];
  char FailFormat[64];
};

std::string autoDetectIsa(llvm::StringRef Path) {
  // Look for ``gfx<digits>[a-z]?`` anywhere in the filename.
  for (size_t I = 0; I + 3 < Path.size(); ++I) {
    if (Path[I] == 'g' && Path[I + 1] == 'f' && Path[I + 2] == 'x') {
      size_t J = I + 3;
      while (J < Path.size() &&
             std::isdigit(static_cast<unsigned char>(Path[J])))
        ++J;
      if (J > I + 3) {
        if (J < Path.size() && Path[J] >= 'a' && Path[J] <= 'z')
          ++J;
        return Path.substr(I, J - I).str();
      }
    }
  }
  return {};
}

int usage() {
  std::fprintf(
      stderr,
      "usage:\n"
      "  raise_cli <code-object.co|.hsaco> [--isa=<arch>] "
      "[--target-isa=<arch>] [--disable-writelane-rewrite] "
      "[--disable-wave-native]\n"
      "  raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] "
      "[--isa=<arch>] [--target-isa=<arch>] "
      "[--disable-writelane-rewrite] [--disable-wave-native]\n"
      "  raise_cli <code-object.co|.hsaco> --write-hsaco=<path> "
      "[--kernel=<name>] [--isa=<arch>] [--target-isa=<arch>] "
      "[--disable-writelane-rewrite] [--disable-wave-native]\n"
      "\n"
      "Default mode: emits per-kernel OK/FAIL lines on stdout in the format\n"
      "  kerneldex coverage expects. Exits 0 iff every kernel raises.\n"
      "--emit-ir mode: dumps raised LLVM IR for a single kernel on stdout.\n"
      "  No fork; stderr left alone for FileCheck.\n"
      "--write-hsaco mode: runs the full pipeline (raise + llc + lld)\n"
      "  for a single kernel and writes the produced HSACO to <path>.\n"
      "  Intended for post-rewrite disassembly triage (see\n"
      "  hotswap/docs/wave-size-translation.md \u00a75.6.3).\n"
      "--target-isa: overrides the target ISA (default: same as --isa).\n"
      "--enable-writelane-rewrite / --disable-writelane-rewrite: controls\n"
      "  the cross-widen-divergent writelane/readlane rewrite (default on;\n"
      "  see wave-size-translation.md \u00a75.6.3). The `--enable-` form is\n"
      "  kept for backward compatibility (existing REWRITE lit RUN lines);\n"
      "  `--disable-` pins the pre-rewrite REFUSE / UNCHANGED path for the\n"
      "  sibling RUN lines. Later-wins on the command line.\n"
      "--enable-wave-native / --disable-wave-native: select between\n"
      "  WaveNativeProjection (post-graduation default) and\n"
      "  ModuloReplicationProjection for wave32 source \u2192 wave64\n"
      "  target cross-widening. The `--enable-` form is kept for\n"
      "  backward compatibility; `--disable-` pins the MODREP path\n"
      "  for lit fixtures and for kernels outside WaveNative's\n"
      "  class coverage (see wave-size-translation.md \u00a7\u00a72.2 / 5.6.1\n"
      "  and modrep-predicate-chain.md \u00a76 for the graduation\n"
      "  rationale). Later-wins on the command line.\n"
      "ISA is inferred from the filename when --isa is not given.\n");
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  std::string CoPath;
  std::string Isa;
  std::string TargetIsa;
  bool EmitIr = false;
  // Default on as of the Triton-corpus graduation (see this file's
  // top-of-file comment and raiser.h for the rationale).  The
  // `--disable-writelane-rewrite` flag (parsed below) forces the
  // pre-rewrite path for the lit fixtures that pin the
  // REFUSE / UNCHANGED sibling contracts.
  bool EnableWritelaneRewrite = true;
  bool EnableWaveNative = true;
  std::string EmitIrKernel;
  std::string WriteHsacoPath;
  std::string WriteHsacoKernel;
  for (int I = 1; I < argc; ++I) {
    std::string A = argv[I];
    if (A.rfind("--isa=", 0) == 0) {
      Isa = A.substr(6);
    } else if (A == "--isa") {
      if (I + 1 >= argc)
        return usage();
      Isa = argv[++I];
    } else if (A.rfind("--target-isa=", 0) == 0) {
      TargetIsa = A.substr(13);
    } else if (A == "--target-isa") {
      if (I + 1 >= argc)
        return usage();
      TargetIsa = argv[++I];
    } else if (A == "--emit-ir") {
      EmitIr = true;
    } else if (A.rfind("--emit-ir=", 0) == 0) {
      EmitIr = true;
      EmitIrKernel = A.substr(10);
    } else if (A.rfind("--write-hsaco=", 0) == 0) {
      WriteHsacoPath = A.substr(14);
    } else if (A.rfind("--kernel=", 0) == 0) {
      WriteHsacoKernel = A.substr(9);
    } else if (A == "--enable-writelane-rewrite") {
      EnableWritelaneRewrite = true;
    } else if (A == "--disable-writelane-rewrite") {
      // Later-wins on the command line: the last occurrence of an
      // --enable- / --disable- pair decides the effective value.  This
      // matches the behaviour every lit fixture's REFUSE / REWRITE RUN
      // lines implicitly rely on (one flag per RUN line).
      EnableWritelaneRewrite = false;
    } else if (A == "--enable-wave-native") {
      EnableWaveNative = true;
    } else if (A == "--disable-wave-native") {
      // Later-wins on the command line, symmetric with
      // --enable-/--disable-writelane-rewrite. Post-graduation the
      // default is on; --disable-wave-native is the opt-out path for
      // lit fixtures that pin MODREP-specific IR shapes (the
      // `cross_wave_warn` warn-only contract, the narrow-O1 C5
      // refusal siblings) and for producer flows that want MODREP's
      // "independent halves" throughput on pointwise kernels. See
      // this file's top-of-file comment.
      EnableWaveNative = false;
    } else if (!A.empty() && A[0] == '-') {
      std::fprintf(stderr, "raise_cli: unknown flag: %s\n", A.c_str());
      return usage();
    } else if (CoPath.empty()) {
      CoPath = A;
    } else {
      std::fprintf(stderr, "raise_cli: unexpected positional arg: %s\n",
                   A.c_str());
      return usage();
    }
  }
  if (CoPath.empty())
    return usage();

  // Read the file up-front so we can fall back to the ELF e_flags
  // ISA when the filename heuristic fails (kerneldex corpora often
  // store kernels under hashed names with no `gfx*` substring; the
  // ELF MACH field is the only deterministic source).
  auto BufOrErr = llvm::MemoryBuffer::getFile(CoPath, /*IsText=*/false);
  if (!BufOrErr) {
    std::fprintf(stderr, "raise_cli: cannot read %s: %s\n", CoPath.c_str(),
                 BufOrErr.getError().message().c_str());
    return 2;
  }
  llvm::StringRef CoStr = (*BufOrErr)->getBuffer();
  std::vector<uint8_t> CoData(CoStr.bytes_begin(), CoStr.bytes_end());

  if (Isa.empty()) {
    Isa = autoDetectIsa(CoPath);
    if (Isa.empty())
      Isa = COMGR::hotswap::detectIsaFromElf(CoData);
    if (Isa.empty()) {
      std::fprintf(stderr,
                   "raise_cli: could not infer ISA from %s; pass --isa=<arch>\n",
                   CoPath.c_str());
      return 2;
    }
  }

  auto KernelNames = COMGR::hotswap::listKernelNames(CoData);
  if (KernelNames.empty()) {
    std::fprintf(stderr, "raise_cli: no kernels in %s\n", CoPath.c_str());
    return 2;
  }

  auto Text = COMGR::hotswap::extractTextSection(CoData);
  if (!Text.Valid) {
    std::fprintf(stderr, "raise_cli: could not extract .text from %s\n",
                 CoPath.c_str());
    return 2;
  }

  // --emit-ir path — no fork, no stderr redirect. Used by lit tests that
  // FileCheck the raised IR on stdout and the raiser diagnostics on
  // stderr. One kernel per invocation.
  if (EmitIr) {
    std::string Target;
    if (EmitIrKernel.empty()) {
      if (KernelNames.size() != 1) {
        std::fprintf(stderr,
                     "raise_cli: --emit-ir requires =<kernel> when the "
                     "code object has %zu kernels\n",
                     KernelNames.size());
        return 2;
      }
      Target = KernelNames.front();
    } else {
      bool Found = false;
      for (const auto &Kn : KernelNames)
        if (Kn == EmitIrKernel) {
          Target = Kn;
          Found = true;
          break;
        }
      if (!Found) {
        std::fprintf(stderr,
                     "raise_cli: kernel '%s' not found in %s\n",
                     EmitIrKernel.c_str(), CoPath.c_str());
        return 2;
      }
    }
    auto Meta = COMGR::hotswap::extractKernelMeta(CoData, Target);
    auto KernelOffsetOrErr = COMGR::hotswap::findKernelSymbolOffset(CoData, Target);
    if (!KernelOffsetOrErr) {
      std::string Err = llvm::toString(KernelOffsetOrErr.takeError());
      std::fprintf(stderr, "raise_cli: kernel '%s' offset lookup failed: %s\n",
                   Target.c_str(), Err.c_str());
      return 1;
    }
    uint64_t KernelOffset = *KernelOffsetOrErr;
    auto Raised = COMGR::hotswap::raiseToIR(Text.Bytes, Isa, Target, Meta,
                                        KernelOffset, TargetIsa,
                                        EnableWritelaneRewrite,
                                        EnableWaveNative);
    if (!Raised.Success) {
      // Contract: raiseToIR only populates RaiseResult::irText on the
      // success path (the last write before setting `success = true`),
      // so we cannot dump partial IR here. Callers that need stderr
      // diagnostics (abort-gate lit tests, etc.) FileCheck the raiser's
      // stderr — we leave that untouched.
      std::fprintf(stderr,
                   "raise_cli: kernel '%s' failed to raise: %s [%s]"
                   " @offset=0x%llx%s%s\n",
                   Target.c_str(),
                   Raised.Failure.Mnemonic.empty()
                       ? "unknown"
                       : Raised.Failure.Mnemonic.c_str(),
                   Raised.Failure.Format.empty()
                       ? "unknown"
                       : Raised.Failure.Format.c_str(),
                   static_cast<unsigned long long>(Raised.Failure.Offset),
                   Raised.Failure.Detail.empty() ? "" : " :: ",
                   Raised.Failure.Detail.empty()
                       ? ""
                       : Raised.Failure.Detail.c_str());
      return 1;
    }
    std::fwrite(Raised.IrText.data(), 1, Raised.IrText.size(), stdout);
    return 0;
  }

  // --write-hsaco path — runs the full pipeline (raise + llc + lld)
  // for a single kernel and writes the resulting HSACO to disk.
  // Triage-mode only: lets downstream tools (llvm-objdump) inspect the
  // exact bytes the gtest harness would launch, so we can walk the
  // Phase 6.5 rewrite end-to-end through the final ISA.
  if (!WriteHsacoPath.empty()) {
    std::string Target;
    if (WriteHsacoKernel.empty()) {
      if (KernelNames.size() != 1) {
        std::fprintf(stderr,
                     "raise_cli: --write-hsaco requires --kernel=<name> when "
                     "the code object has %zu kernels\n",
                     KernelNames.size());
        return 2;
      }
      Target = KernelNames.front();
    } else {
      bool Found = false;
      for (const auto &Kn : KernelNames)
        if (Kn == WriteHsacoKernel) {
          Target = Kn;
          Found = true;
          break;
        }
      if (!Found) {
        std::fprintf(stderr,
                     "raise_cli: kernel '%s' not found in %s\n",
                     WriteHsacoKernel.c_str(), CoPath.c_str());
        return 2;
      }
    }
    std::string EffectiveTargetIsa = TargetIsa.empty() ? Isa : TargetIsa;
    auto Pipe = COMGR::hotswap::runPipeline(CoData, Isa, EffectiveTargetIsa,
                                        Target, EnableWritelaneRewrite,
                                        EnableWaveNative);
    if (!Pipe.Success) {
      std::fprintf(stderr,
                   "raise_cli: pipeline failed for kernel '%s' (lifted=%d/%d, "
                   "failMnemonic='%s')\n",
                   Target.c_str(), Pipe.LiftedCount, Pipe.TotalCount,
                   Pipe.FailMnemonic.c_str());
      return 1;
    }
    FILE *Fp = std::fopen(WriteHsacoPath.c_str(), "wb");
    if (!Fp) {
      std::fprintf(stderr, "raise_cli: cannot open %s for writing\n",
                   WriteHsacoPath.c_str());
      return 2;
    }
    size_t Wrote =
        std::fwrite(Pipe.Hsaco.data(), 1, Pipe.Hsaco.size(), Fp);
    std::fclose(Fp);
    if (Wrote != Pipe.Hsaco.size()) {
      std::fprintf(stderr,
                   "raise_cli: short write to %s (%zu of %zu bytes)\n",
                   WriteHsacoPath.c_str(), Wrote, Pipe.Hsaco.size());
      return 2;
    }
    std::fprintf(stderr,
                 "raise_cli: wrote %zu byte HSACO for kernel '%s' to %s "
                 "(lifted %d/%d)\n",
                 Pipe.Hsaco.size(), Target.c_str(), WriteHsacoPath.c_str(),
                 Pipe.LiftedCount, Pipe.TotalCount);
    return 0;
  }

  int TotalKernels = 0, OkKernels = 0, FailKernels = 0, CrashKernels = 0;

  for (auto &KName : KernelNames) {
    ++TotalKernels;

    auto KernelOffsetOrErr = COMGR::hotswap::findKernelSymbolOffset(CoData, KName);
    if (!KernelOffsetOrErr) {
      std::string Err = llvm::toString(KernelOffsetOrErr.takeError());
      std::fprintf(stderr, "raise_cli: kernel '%s' offset lookup failed: %s\n",
                   KName.c_str(), Err.c_str());
      ++FailKernels;
      std::printf("FAIL %s -> __kernel_offset__ "
                  "[KernelSymbolOffsetLookupFailed]\n",
                  KName.c_str());
      continue;
    }
    uint64_t KernelOffset = *KernelOffsetOrErr;

#if defined(LLVM_ON_UNIX)
    // Crash-isolated path: each kernel runs in a forked child writing its
    // result into a MAP_ANONYMOUS shared region the parent then drains.
    auto *Shm = static_cast<KernelRaiseStats *>(
        mmap(nullptr, sizeof(KernelRaiseStats), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (Shm == MAP_FAILED) {
      std::fprintf(stderr, "raise_cli: mmap failed\n");
      return 3;
    }
    std::memset(Shm, 0, sizeof(KernelRaiseStats));

    pid_t Pid = fork();
    if (Pid == 0) {
      // Silence the child's stderr: LLVM chatters a lot, and kerneldex
      // only cares about OK/FAIL on stdout plus the last stderr line
      // when the process as a whole crashes.
      int Devnull = open("/dev/null", O_WRONLY);
      if (Devnull >= 0) {
        dup2(Devnull, STDERR_FILENO);
        close(Devnull);
      }
      auto Meta = COMGR::hotswap::extractKernelMeta(CoData, KName);
      auto Raised = COMGR::hotswap::raiseToIR(Text.Bytes, Isa, KName, Meta,
                                          KernelOffset, TargetIsa,
                                          EnableWritelaneRewrite,
                                          EnableWaveNative);
      Shm->Done = true;
      Shm->Success = Raised.Success;
      Shm->Lifted = Raised.LiftedCount;
      Shm->Total = Raised.TotalCount;
      if (!Raised.Success) {
        const char *Mn = Raised.Failure.Mnemonic.empty()
                             ? "unknown"
                             : Raised.Failure.Mnemonic.c_str();
        const char *Fmt = Raised.Failure.Format.empty()
                              ? "unknown"
                              : Raised.Failure.Format.c_str();
        std::strncpy(Shm->FailMnemonic, Mn, sizeof(Shm->FailMnemonic) - 1);
        std::strncpy(Shm->FailFormat, Fmt, sizeof(Shm->FailFormat) - 1);
      }
      _exit(0);
    }

    int St = 0;
    waitpid(Pid, &St, 0);

    if (!Shm->Done || WIFSIGNALED(St) || (WIFEXITED(St) && WEXITSTATUS(St) != 0)) {
      // Child never wrote the shm marker, or died by signal, or exited
      // with a nonzero status: surface this as a FAIL row with a
      // synthetic mnemonic so kerneldex still counts the kernel.
      ++CrashKernels;
      int Sig = WIFSIGNALED(St) ? WTERMSIG(St) : 0;
      std::printf("FAIL %s -> __crash__ [signal_%d]\n", KName.c_str(), Sig);
    } else if (Shm->Success) {
      ++OkKernels;
      std::printf("OK %s (%d/%d)\n", KName.c_str(), Shm->Lifted, Shm->Total);
    } else {
      ++FailKernels;
      std::printf("FAIL %s -> %s [%s]\n", KName.c_str(), Shm->FailMnemonic,
                  Shm->FailFormat);
    }

    munmap(Shm, sizeof(KernelRaiseStats));
#else
    // Windows fallback: no fork, so raise in-process. A crash here aborts
    // the whole tool instead of becoming a FAIL row, but per-kernel OK/FAIL
    // accounting is otherwise identical to the Unix path.
    auto meta = COMGR::hotswap::extractKernelMeta(coData, kName);
    auto raised = COMGR::hotswap::raiseToIR(text.Bytes, isa, kName, meta,
                                            kernelOffset, targetIsa,
                                            enableWritelaneRewrite,
                                            enableWaveNative);
    if (raised.success) {
      ++okKernels;
      std::printf("OK %s (%d/%d)\n", kName.c_str(), raised.liftedCount,
                  raised.totalCount);
    } else {
      ++failKernels;
      const char *mn = raised.failure.mnemonic.empty()
                           ? "unknown"
                           : raised.failure.mnemonic.c_str();
      const char *fmt = raised.failure.format.empty()
                            ? "unknown"
                            : raised.failure.format.c_str();
      std::printf("FAIL %s -> %s [%s]\n", kName.c_str(), mn, fmt);
    }
#endif
  }

  std::fprintf(stderr,
               "raise_cli: %d kernels, %d ok, %d fail, %d crash (%s)\n",
               TotalKernels, OkKernels, FailKernels, CrashKernels,
               CoPath.c_str());

  return (FailKernels + CrashKernels) == 0 ? 0 : 1;
}
