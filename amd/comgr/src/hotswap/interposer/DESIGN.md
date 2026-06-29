# HotSwap device-spoofing transpilation interposer

**Status:** bring-up. Base: clean head of the `hotswap` branch (`613e1750c740`).

## Goal

Make HotSwap transpilation *complete by construction*. Present a spoofed **gfx1250**
device to the entire ROCm stack so every library and runtime emits gfx1250 code
objects. Capture every code object, transpile gfx1250 -> the **real** device ISA
(gfx942 / gfx950 / gfx1151, auto-detected), and run the transpiled result on the
real hardware. By construction nothing native reaches the GPU and we observe every
dispatch.

This is the *active* mechanism the read-only completeness checker
(`amd/comgr/src/hotswap/audit/`, separate branch) was built to motivate.

## ISA model

- **Spoof / source = gfx1250, always.** The whole stack is shown gfx1250 and emits
  gfx1250 code objects.
- **Execution target = the real hardware**, auto-detected at runtime
  (gfx942 / gfx950 / gfx1151). On this dev box it is gfx1151.
- **Transpile is always gfx1250 -> <real detected target>.**

## One component, two in-process seams

The interposer ships as a single library, `libhotswap-interposer.so`, registered two
ways and hooking two seams in the same process:

1. **KFD / sysfs seam (`LD_PRELOAD`).** Detect the *real* device ISA first, by reading
   the true KFD topology before any spoof is installed -- this is the only place the
   real target is knowable, because once gfx1250 is presented upward even ROCr's own
   `hsa_isa_get_info_alt` returns gfx1250. Then present gfx1250 to the stack by
   redirecting reads of the KFD topology `properties` file and overriding the single
   `gfx_target_version` field; all real `/dev/kfd` and `/dev/dri` traffic is forwarded
   untouched to the real driver, so kernels actually run on the real device. Own queue
   creation + doorbells as the completeness chokepoint.
2. **Code-object seam (`HSA_TOOLS_LIB`).** Capture the host gfx1250 ELF at the HSA
   `CoreApiTable` code-object load entry points, transpile gfx1250 -> real via
   `amd_comgr_hotswap_transpile_with_options`, and substitute the transpiled, real-ISA
   object. **Captured at load, not at the queue**: the ELF only exists in host-readable
   form at load time. By the time work reaches the queue, the runtime has already copied
   the *loaded image* (not the ELF) into GPU memory, which on a discrete GPU
   (gfx942/gfx950) is not host-readable. This is why capture cannot live purely at the
   KFD boundary.
3. **Completeness backstop (doorbell / AQL).** At queue submit, decode each AQL
   `KERNEL_DISPATCH` (`kernel_object` at byte offset 32) and assert its code object
   passed through transpilation; refuse (production) or loud-log (audit) otherwise.

## Why the surgical `gfx_target_version` override

`libhsakmt/src/topology.c` reads sysfs `gfx_target_version` into `props->EngineId`
(`topology.c:1256`), and that `EngineId` is what the whole stack queries as the device
ISA. libhsakmt opens the node `properties` file by absolute path via `fopen`
(`get_topology_dir()` + `topology_sysfs_get_node_props`), so redirecting that one file
to a copy with a single edited field is sufficient to make CLR / rocBLAS / MIOpen /
Triton / ROCr select and emit gfx1250. Every other topology field is passed through with
its real value, so the runtime configures queues, scratch, LDS and apertures for the
real hardware that actually executes the transpiled code.

### Expected bonus

Because the device genuinely *reports* gfx1250, the CLR fatbin-selection and ISA-compat
gate patches (the documented "real shipping blocker") should become unnecessary: fatbin
selection picks the gfx1250 slice naturally and `amd::Isa::isCompatible` passes
(agent == code object == gfx1250). To be confirmed during bring-up.

## Known risks (validate on box; some are gfx1151-only assumptions)

- `topology.c:1344-1345` derives `SGPRSizePerCU` / `VGPRSizePerCU` from the (spoofed)
  `EngineId`, and `topology.c:2307` has a `GFX_VERSION_GFX1151`-specific mem-bank path.
  A few execution-setup capabilities are therefore derived from the spoofed version
  rather than from the passed-through fields. Mitigation order: surgical field-only
  spoof first; if queue/scratch setup misbehaves, pin the derived caps to real values.
- dGPU specifics not validated here: device-VRAM non-readability, doorbell GPUVM
  mapping, multi-XCC / multi-node topology, and the spoofed-topology specifics for CDNA.

## Reuse

- KFD/sysfs libc interposition adapts ROCm `emulation/rocjitsu`'s
  `kmd/linux/interposer.cpp` (MIT) -- the `LD_PRELOAD` hook skeleton, the glibc
  `stat`/`open`/`fcntl64` variant handling, the `dlsym(RTLD_NEXT)` passthrough table,
  the reentrancy guard, and `fork` reset. rocjitsu *emulates* the device, so its
  simulated/remote drivers, synthetic DRM, and synthetic-topology generation are
  dropped; the passthrough design keeps the real `/dev/kfd` and `/dev/dri`. Files
  derived from rocjitsu keep their MIT headers.
- Code-object capture + COMGR transpile call adapt the HotSwap HSA tool lib
  (`rocm-systems/projects/hotswap`).
- The doorbell / AQL backstop adapts the audit lib (`amd/comgr/src/hotswap/audit/`).

## Configuration (env, default-off)

The interposer is inert unless explicitly enabled, since it is injectable via
`LD_PRELOAD`.

- `HOTSWAP_INTERPOSER_SPOOF` -- gfx target to present upward, e.g. `gfx1250` or
  `125000`. Unset => fully inert passthrough (no redirect, no spoof).
- `HOTSWAP_INTERPOSER_TARGET` -- optional override of the real transpile target;
  default is auto-detected from the true topology.
- `HOTSWAP_INTERPOSER_LOG` -- `1` to emit one-line diagnostics to stderr.

## Runtime requirement: a gfx1250-aware ROCr

Pushing the spoof to the KFD layer forces ROCr to construct a *gfx1250 agent*, which
requires ROCr's ISA registry to know gfx1250. This is a new requirement that the
load-time hotswap tool lib avoids (it keeps the real gfx1151 agent and only
string-spoofs the ISA *name* via `hsa_isa_get_info_alt`). ROCr's registry is a
hardcoded table in `runtime/hsa-runtime/core/runtime/isa.cpp` (`ISAREG_ENTRY_GEN`).
gfx1250 is present in the in-tree rocr-runtime
(`isa.cpp`: `ISAREG_ENTRY_GEN("gfx1250", 12, 5, 0, ..., 32, "gfx12-generic")`) but
*not* in the installed `/opt/rocm` ROCr on this box. So the interposer must run
against the locally built ROCr, e.g.:

```
LD_LIBRARY_PATH=<rocm-systems>/rocr-install/lib \
HOTSWAP_INTERPOSER_SPOOF=gfx1250 \
LD_PRELOAD=<build>/libhotswap-interposer.so \
rocminfo
```

Verified: with the gfx1250-aware ROCr the GPU agent reports `gfx1250` /
`amdgcn-amd-amdhsa--gfx1250` while executing on the real gfx1151; against an older
ROCr that lacks the entry the agent is dropped (degraded `ISA Info:`).

## Phases

- **Phase 1** -- KFD/sysfs identity spoof + passthrough. Gate: `rocminfo` reports
  gfx1250 while running on the real gfx1151. **Done.**
- **Phase 2** -- load-time capture + COMGR transpile + substitute. Gate: a single HIP
  kernel runs with correct results, captured and transpiled, nothing native dispatched.
- **Phase 3** -- doorbell/AQL backstop. Gate: zero un-transpiled dispatches.
- **Phase 4** (follow-on) -- SGLang Qwen3-0.6B end to end.
