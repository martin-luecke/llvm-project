# rocPRIM decoupled look-back scan under WaveNative: a structural limitation

## Summary

`torch.cumsum` (and any rocPRIM device scan) over a large array faults under the
gfx1250 -> gfx942 hotswap transpiler. The fault is an out-of-bounds global load
in rocPRIM's decoupled look-back scan. It is **not** a localized handler bug: it
is a structural incompatibility between the kernel's per-warp *scalar* look-back
loop and the WaveNative projection's packing of two source wave32 warps into one
wave64 that shares a single SGPR file.

This document records the diagnosis (with the evidence), why none of the existing
projections resolve it, and the realistic options.

## Reproducer

Deterministic, minimal, no model required. With the rocPRIM trampoline name-shim
active (so dispatch reaches the real generic-arch scan kernel instead of the
gfx942 `__builtin_unreachable` stub -- see "Dispatch" below) and no other
workarounds:

```python
import torch, sys
sys.path.insert(0, "/workspace/rocm-hotswap-testing/runtime/container_smokes")
from hotswap_smoke_common import init_torch_with_gfx_override as f
f(torch)
x = torch.rand(1, 151936, device="cuda")
torch.cumsum(x, dim=-1)          # <-- Memory access fault (HSA_STATUS_ERROR_MEMORY_FAULT)
torch.cuda.synchronize()
```

Run under `LD_PRELOAD=/opt/hotswap/lib/libhip_device_name_override.so`.
Threshold: works for n <= 65536 (single tile, no decoupled look-back), faults for
n = 151936 (multi-tile, look-back taken). The failing kernel is
`rocprim...scan_impl...lookback_scan_determinismE0...target_archE4294967295`
(the generic-arch trampoline, which holds the real code).

## What was ruled out (verified, not assumed)

Each localized suspect was checked against the source and the emitted IR:

- **Memory-op guard.** `RaiseContext::emitUnderExec` gates every guarded global
  op on `Projection.emitLaneActiveBit(Regs.loadExec())` -- the true modeled
  source EXEC alloca, not a broadcast/OR-ed shadow. The guard is correct; the
  faulting lane is *legitimately* active per the modeled EXEC.
- **`v_readfirstlane`.** `handle-valu-cross-lane.cpp` lowers it source-wave-scoped
  under cross-widening: it computes the per-source-wave EXEC slice
  (`loadExec >> (laneId & ~(W_src-1))`), takes `cttz` for the first active lane
  within that slice, and gathers with `ds_bpermute`. Correct.
- **Address arithmetic / zext vs sext.** The garbage address decodes as
  `base + zext(i32 -28) * 8`. Sign-extending would only change the crash into an
  in-range read of adjacent garbage (still wrong data). The offset is a faithful
  translation of the source; the real problem is that the index reached -28 at
  all.
- **The clamp fix (`v_{add,sub,subrev}_nc_u32` clamp).** Unrelated here: the scan
  look-back decrements with a signed scalar `s_add_co_i32 s2, s2, -1`, not the
  clamped `v_sub_nc_u32` that rocPRIM's merge_path uses.

## Root cause (proven from the source disassembly)

Disassembling the source gfx1250 look-back loop shows it is driven entirely by
scalar registers:

```
s_add_co_i32 s2, s2, -1            ; loop counter lives in an SGPR
s_cmp_lg_u32 s25, s2 / s_cbranch_vccz
s_cmp_lg_u32 s25, 0  / s_cbranch_scc0   ; scalar-condition loop termination
v_readfirstlane_b32 s3, v1        ; predecessor tile value broadcast to a scalar
```

So the decoupled look-back is a **per-warp scalar loop**: its counter (`s2`) and
control flow (SCC/VCC branches) are uniform-per-wave SGPR state.

WaveNative packs two source wave32 warps into one wave64, and the two packed
warps **share the single wave64 SGPR file**. When the two warps diverge -- warp 0
has found its inclusive prefix and stopped, warp 1 is still walking back -- there
is only one `s2` for both. Live evidence at the fault: `EXEC = 0xffffffff00000000`
(source warp 1 active on target lanes 32-63, warp 0 masked on 0-31), and the
per-lane load address `v[44:45]` is the uniform value `0x8002aa328b20`
= `base + (-28) * 8`. The single shared counter marched the still-active warp 28
tiles past tile 0, producing an out-of-bounds descriptor address.

Two packed warps cannot hold divergent scalar loop state in a shared SGPR file.
This is the structural limit.

## Why no existing projection resolves it

The kernel needs both properties at once:
- **Concurrent warps** -- the block-level scan phase uses LDS + `s_barrier`, so
  the two source warps must run together.
- **Independent scalar state** -- the look-back phase has a per-warp scalar loop
  whose counter must be private to each source warp.

The projections each provide only one:

| Projection | Concurrent warps | Independent SGPRs | Applies to a 256-thread LDS+barrier kernel? |
|------------|:----------------:|:-----------------:|:-------------------------------------------:|
| WaveNative (current) | yes | **no** (shared file) | yes -- and hits the collision |
| ThreadLoop (serialize waves) | **no** | yes | **no** -- disallowed for LDS/barrier kernels (serialization breaks barrier sync) |
| ModuloReplication (own wave64 per source wave) | yes | yes | **no** -- only valid for < 64-thread workgroups; 256 threads would need doubling the workgroup size, which hotswap does not do |

There is no projection that gives concurrent + independent-scalar at this
workgroup size.

## Dispatch note (separate, already-understood issue)

Independent of the scan bug: rocPRIM 4.2 wraps kernels in per-arch
`trampoline_kernel`s and picks one from `hipDeviceProp.name` ("AMD Instinct
MI300X" -> gfx942). The gfx1250-compiled binary puts real code only in the
generic (arch = -1) trampoline; the gfx942 slot is a `__builtin_unreachable`
stub. So without intervention the sort/scan dispatch a no-op stub. Neutralizing
`prop.name` (the `libhip_device_name_override.so` LD_PRELOAD shim) makes rocPRIM
fall back to the generic real-code trampoline. This is a rocPRIM host-dispatch
quirk, not a transpiler bug, and the shim is an acceptable environment fix. It is
also *required* to even reach the scan bug above (otherwise the stub just no-ops).

## What does NOT count as a fix

`torch.use_deterministic_algorithms(True)` was tried and initially reported as a
solution. It is **not**: `cumsum` is not a torch-nondeterministic op, so the flag
has no documented reason to change its kernel, and the fault is GPU-memory-state
dependent -- the flag most likely just perturbed allocation/workspace state and
incidentally dodged the OOB. It is masking, not a fix, and it is the kind of
environment workaround explicitly disallowed for this work. Do not present it as a
solution.

## Potential next steps (all non-trivial; none are the det-mode hack)

1. **Re-vectorize the divergent scalar loop in WaveNative (the true transpiler
   fix).** Detect that a scalar loop's counter/condition can diverge between the
   two packed source warps, and promote that SGPR state to a per-source-wave
   representation (e.g. a value that differs across the two 32-lane halves),
   converting the scalar SCC/VCC control flow into vector/predicated control flow
   masked per source wave. This is the only path that keeps concurrency (LDS/
   barriers intact) while giving each warp private loop state. It is a
   substantial change to scalar-control-flow modeling and must be validated
   against the cumsum repro (fault -> pass, name-shim only, no flags) and full
   forward parity (f32 40/40, bf16 30/30). High risk; touches core EXEC/SGPR
   handling used by every kernel.

2. **rocPRIM-side: build with the deterministic (reduce-then-scan) path for this
   device.** The deterministic scan has no divergent per-warp scalar look-back,
   so it avoids the structural collision entirely. Unlike the runtime
   `use_deterministic_algorithms` flag (masking), selecting the deterministic
   scan algorithm at rocPRIM build/config time is a legitimate, explainable
   change. Requires a rocPRIM build knob or patch, outside comgr.

3. **Scoped serialization of only the barrier-free look-back sub-region.** If the
   decoupled look-back phase is barrier-free (the barriers live in the earlier
   block-scan phase), the transpiler could serialize just that sub-region across
   the two packed warps while keeping the barrier-bearing phase concurrent. This
   sidesteps ThreadLoop's global "no LDS/barrier" restriction but requires
   identifying the sub-region and is sophisticated.

4. **Loudly refuse the kernel instead of miscompiling.** If none of the above is
   done, the transpiler should at least detect the divergent-per-warp-scalar-loop
   shape and refuse (RaiseFailure) rather than emit code that OOB-faults at
   runtime. This is the "fail loudly, never silently" fallback: correct-by-
   refusal, and it turns a GPU memory fault into a diagnosable lift-time error.

## Validation harness

- Fault repro: `torch.cumsum(rand(1,151936))` + name-shim, no det-mode -> must go
  from Memory access fault to correct output for any claimed fix.
- Non-regression: pure-argmax greedy decode of qwen3-0.6b must stay bit-identical
  to CPU (f32 40/40, bf16 30/30 tokens), and the hotswap-raise lit suite must stay
  green -- to prove a scalar-control-flow change did not break the (already
  correct and merged) forward path.
