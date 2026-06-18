# LDS → Global-Memory Redirect

> **Status:** prototype implementation, gated behind `HSA_HOTSWAP_LDS_TO_GLOBAL=1`.
> Default behaviour (refuse when `GroupSegmentFixedSize > LdsByteCapacity`) is unchanged.
> All sections marked **[STUB]** are explicitly refused at raise time with a
> `RaiseFailure::ldsGlobalRedirectUnsupportedVariant` error; no silent miscompile.

---

## 1. Why this exists

gfx1250 exposes up to 320 KB of LDS per workgroup
(`programming_manual_1250.txt:18306`).  gfx1151 caps LDS at 64 KB
(`isa-profile.h:96`, `LdsByteCapacity = 65536`).  A kernel whose
`GroupSegmentFixedSize` exceeds 65536 bytes cannot run on gfx1151 under the
normal refuse gate (`abi-translation.md §7 G1`).  This transform replaces
that refuse, behind an opt-in flag, by routing every storage DS op to a
per-workgroup region of unified global memory.

**Inviolable constraint:** we operate on a compiled code object.  The source
cannot be recompiled.  Every "just use smaller LDS" answer is out of scope.

---

## 2. Capacity gap

| Target | LDS / workgroup | Source |
|---|---|---|
| gfx1250 (source) | up to 320 KB | `programming_manual_1250.txt:18306` |
| gfx1151 (target) | 64 KB (`LdsByteCapacity = 65536`) | `isa-profile.h:96` |

---

## 3. The transform: what changes in each kernel

### 3.1 New kernarg: `wg_lds_base`

A new `ptr addrspace(1)` argument is appended to the lifted function signature.
It carries the base address of a contiguous global-memory allocation sized
`num_workgroups × G` bytes, where `G = GroupSegmentFixedSize`.  The per-WG
byte region for workgroup `w` is `[wg_lds_base + w*G, wg_lds_base + (w+1)*G)`.

The lifted kernel's kernarg segment grows by 8 bytes (one pointer).  The
interceptor/launch side is responsible for populating this slot before
dispatch (§6).

### 3.2 Per-workgroup base pointer, computed once at kernel entry

At the beginning of the lifted kernel:

```
blocks_x = ceil(grid_x / wg_size_x)   (= grid_x / wg_size_x, integer div from dispatch packet)
blocks_y = ceil(grid_y / wg_size_y)

wg_id_x = llvm.amdgcn.workgroup.id.x()
wg_id_y = llvm.amdgcn.workgroup.id.y()
wg_id_z = llvm.amdgcn.workgroup.id.z()

linear_wg_id = (wg_id_z * blocks_y + wg_id_y) * blocks_x + wg_id_x
wg_base      = wg_lds_base + linear_wg_id * G          (ptr addrspace(1))
```

`wg_base` is an SGPR-pair (uniform within the workgroup, divergent between
workgroups).  Stored in `RaiseContext::WgLdsBase` and used by every redirected
DS op in the same kernel.

Grid dimensions come from `llvm.amdgcn.dispatch.ptr` loads (same as
`source-hidden-args.cpp`'s `emitHiddenBlockCount`).

### 3.3 Storage DS ops → global loads/stores

For every single-offset storage DS op (`IsDsRead` / `IsDsWrite` in
`handle-ds.cpp:584`), instead of:

```cpp
Value *Ptr = Ctx.B.CreateIntToPtr(Addr, PointerType::get(Ctx.C, 3)); // AS3 = LDS
```

emit:

```cpp
Value *BaseI64 = Ctx.B.CreatePtrToInt(Ctx.WgLdsBase, Ctx.I64Ty);
Value *EffAddr  = Ctx.B.CreateAdd(BaseI64, Addr);           // Addr is already i64
Value *Ptr      = Ctx.B.CreateIntToPtr(EffAddr, PointerType::get(Ctx.C, 1)); // AS1 = global
```

The load/store IR nodes (and their alignment, vector type) are otherwise
identical to the LDS path.  Because the address space changed from 3 to 1,
the AMDGPU backend selects `global_load_*` / `global_store_*` (VMEM path)
rather than `ds_load_*` / `ds_store_*` (LDS path).

### 3.4 LDS size attribute

When the redirect is active the lifted kernel does not use LDS at all (only
non-storage DS ops like `ds_bpermute` / `ds_swizzle` remain, but those use the
LDS crossbar without storage allocation).  The `amdgpu-lds-size` function
attribute is omitted (or set to 0) so the hardware allocates no LDS for the
redirected kernel.

---

## 4. Barrier / memory-model treatment (correctness-critical)

### 4.1 Why plain `s_barrier` is insufficient

On gfx1151, `s_barrier` provides workgroup rendezvous (all waves arrive before
any leave) but **does not flush or invalidate the L0 vector cache**.  Each SIMD
within a CU has its own 16 KB L0; stores from wave A on SIMD-0's L0 are not
automatically visible to wave B on SIMD-1's L0 without explicit cache actions.

Under LDS, the hardware cache-coherence for `ds_store` / `ds_load` is
guaranteed within the workgroup by the LDS hardware itself — no extra
instructions are needed around `s_barrier`.  Under global memory, this
guarantee is absent.

**Consequence:** every `s_barrier` that protects a redirected DS read/write
pair must be upgraded to:

```
fence syncscope("workgroup-one-as") release   ; drain global store buffer (→ s_wait_storecnt 0)
s_barrier                                      ; workgroup rendezvous
fence syncscope("workgroup-one-as") acquire   ; invalidate L0 (→ buffer_gl0_inv)
```

The AMDGPU backend lowers these IR fences as:
- `release` → `s_wait_storecnt 0x0` (wait for outstanding global stores)
- `acquire` → `buffer_gl0_inv` (invalidate the L0 vector cache)

Justification from the hardware specification:
- `s_barrier` semantics: workgroup synchronization only, no cache flush
  (sp3 GFX11.5 ISA manual, see §2.5 DS Instructions; no cache-coherence note)
- `buffer_gl0_inv` semantics: invalidates the L0 vector cache for the issuing
  CU, ensuring subsequent loads fetch from L1/L2 rather than stale L0 data
  (sp3 GFX11.5 ISA manual, BUFFER section)
- `s_wait_storecnt`: drains the VMEM store count to ≤ N; at 0, all VMEM stores
  have left the per-wave store buffer and are visible to other SIMDs on the
  same CU via L1 (sp3 GFX11.5, S_WAIT_STORECNT)

### 4.2 Implementation in handle-sopp.cpp

The barrier handler (`handle-sopp.cpp`, `CanonicalOp::S_BARRIER` /
`S_BARRIER_WAIT`) gains:

```cpp
if (Ctx.LdsGlobalRedirect) {
  SyncScope::ID WgOneAs =
      Ctx.C.getOrInsertSyncScopeID("workgroup-one-as");
  Ctx.B.CreateFence(AtomicOrdering::Release, WgOneAs);
}
auto *BarrierFn =
    Intrinsic::getOrInsertDeclaration(&Ctx.M, Intrinsic::amdgcn_s_barrier);
Ctx.B.CreateCall(BarrierFn, {});
if (Ctx.LdsGlobalRedirect) {
  SyncScope::ID WgOneAs =
      Ctx.C.getOrInsertSyncScopeID("workgroup-one-as");
  Ctx.B.CreateFence(AtomicOrdering::Acquire, WgOneAs);
}
```

**Conservative policy:** the fences are emitted around every barrier in the
kernel when `LdsGlobalRedirect` is active.  A per-barrier "does this barrier
protect any redirected DS op" analysis is not done in the prototype.  The
overhead is one extra `s_wait_storecnt` + one `buffer_gl0_inv` per barrier; for
kernels that use barriers primarily for LDS communication this is the common
case anyway.

### 4.3 Wait-counter treatment

Source `s_wait_dscnt` / `s_wait_loadcnt_dscnt` instructions that were ordering
DS operations are dropped by the existing no-op waitcnt handler
(`handle-sopp.cpp:118-127`) — the AMDGPU backend inserts the correct
`s_wait_storecnt` for the global stores via the IR memory model.  No change
needed here.

### 4.4 `S_BARRIER_SIGNAL` (gfx12 split-barrier)

The source may emit `s_barrier_signal` / `s_barrier_wait` pairs (gfx12 split
form) instead of the legacy unified `s_barrier`.  The existing handler maps
`S_BARRIER_WAIT` → `llvm.amdgcn.s.barrier` and treats `S_BARRIER_SIGNAL` as a
no-op.  The fence upgrade applies to the `S_BARRIER_WAIT` site (where the
`llvm.amdgcn.s.barrier` call is emitted).

---

## 5. Cross-lane DS ops — what we do NOT redirect

The following DS ops use the **LDS crossbar** but allocate **no LDS storage**.
They must never be redirected; redirecting them would produce a global memory
address used as a permutation selector, which is semantically incorrect.

| Op | Handle site | Why excluded |
|---|---|---|
| `DS_BPERMUTE_B32` | `handle-ds.cpp:757` | cross-lane gather via selector, no storage |
| `DS_SWIZZLE_B32` | `handle-ds.cpp:814` | fixed-pattern swizzle, no storage |
| `DS_PERMUTE_B32` (gfx1250) | `handle-valu-cross-lane.cpp` | forward permute, no storage |

These ops already have dedicated handler blocks before the `IsDsRead`/`IsDsWrite`
generic block and cannot be confused with storage ops.  The split is verified
in `hotswap-lds-redirect-xlan.s` (§9).

---

## 6. Storage variants in scope and out of scope for the prototype

### 6.1 Redirected (prototype)

| Family | `CanonicalOp` range | Comment |
|---|---|---|
| Single-offset reads | `DS_READ_B{8,16,32,64,96,128}`, `DS_READ_{I8,U8,I16,U16}` | `IsDsRead` block in `handle-ds.cpp` |
| Single-offset writes | `DS_WRITE_B{8,16,32,64,96,128}` | `IsDsWrite` block |

### 6.2 Refused when redirect is active (explicit `ldsGlobalRedirectUnsupportedVariant`)

| Family | Why |
|---|---|
| DS2 (`DS_READ2_*/DS_WRITE2_*`) | Two-offset semantics; addressing is unit-scaled, non-trivial to re-map; low priority in corpus |
| D16_HI variants (`DS_WRITE_B16_D16_HI`, `DS_WRITE_B8_D16_HI`) | Partial-VGPR semantics; not observed in overflow corpus kernels |
| DS_ADD_F64 (atomic) | LDS atomic; no direct global equivalent with identical memory-model guarantees without CAS loop |
| All other DS atomics | Same: staged for future work |
| TR8/TR16 transpose loads | Use LDS + cross-lane; redirect would break transpose semantics |
| Async-to-LDS (`GLOBAL_LOAD_ASYNC_TO_LDS_*`, `DS_DIRECT_LOAD`) | Async DMA ops; redirect semantics are ill-defined for the async unit |

A refused variant returns `RaiseFailure::ldsGlobalRedirectUnsupportedVariant`
with the specific mnemonic.  This is strictly better than a silent miscompile.
Coverage can be staged: add DS2 next (straightforward two-independent-GEPs
pattern, parallel to the existing two-offset handler), then DS atomics via CAS
loop.

---

## 7. Scaling caveat

The prototype allocates `num_workgroups × G` bytes of global memory, where:

```
num_workgroups = ceil(grid_x / wg_x) × ceil(grid_y / wg_y) × ceil(grid_z / wg_z)
```

For a 2D matmul grid with 256×256 workgroups and G = 96 KB (a typical
Triton-large kernel), this is 256 × 256 × 96 KB ≈ 6 GB.  That fits within a
typical 24–40 GB VRAM budget but is wasteful: at any given moment only a small
fraction of the workgroups are resident on the chip.

**Bounded recycling pool design (sketch, not implemented):**

Allocate a pool of size `num_resident_wgs × G` where `num_resident_wgs` ≈
`num_CUs × max_wgs_per_CU`.  Each workgroup atomically claims a slot from the
pool at kernel entry (`atomicrmw add` on a counter mod `pool_size`) and releases
it at exit.  The slot index replaces `linear_wg_id` in the `wg_base`
computation.  This caps the allocation at `num_resident_wgs × G` regardless of
grid size.  For gfx1151 (20 CUs, ~4 WGs/CU at high occupancy), the cap is
roughly `80 × G` — 7.5 MB for G = 96 KB.

The prototype does not implement this to avoid complexity; the full-grid
allocation is logged at raise time with a cap estimate.

---

## 8. Launch-side contract

The interception library (`libhotswap_intercept.so`) must:

1. **Detect redirect** by reading the `hotswap.lds_redirect_size` function
   metadata from the translated code object (value = `G` in bytes; zero means
   no redirect).
2. **Allocate** `num_workgroups × G` bytes of coarse-grained global memory via
   `hsa_memory_allocate` (or `hipMalloc`).
3. **Extend the kernarg buffer** by 8 bytes beyond the source kernarg size.
4. **Copy** the original kernarg bytes verbatim.
5. **Write** the allocation base pointer at byte offset
   `original_kernarg_segment_size` in the extended buffer.
6. **Set** `kernarg_segment_size` in the AQL dispatch packet to
   `original_size + 8`.
7. **Dispatch** as normal.
8. **Free** the global allocation after `hsa_queue_packet_acquire` confirms
   completion (or lazily via a reusable pool; see §7).

The kernel reads the pointer via a GEP at `kernarg_segment_size - 8` from
`amdgcn_kernarg_segment_ptr`, typed as `ptr addrspace(1)`.

**Metadata key:** `hotswap.lds_redirect_size` is attached to the translated
`llvm::Function` as `!{i64 G}`.  The interceptor reads it via the COMGR API's
metadata access path or via a dedicated ELF note (TBD; for the prototype the
metadata survives in the YAML `.args` as an extra arg with value_kind
`"hotswap_lds_base"`).

---

## 9. A/B test plan

### 9.1 Primary oracle — force-redirect of a kernel that fits in 64 KB

A kernel whose `GroupSegmentFixedSize ≤ 64 KB` is translated twice:
(a) normal path (DS stays DS), (b) with `HSA_HOTSWAP_LDS_TO_GLOBAL_FORCE=1`.
Both run on gfx1151.  **Require bit-identical outputs.**

Test cases:
1. **Single-wave producer/consumer**: one wave writes to DS, reads back.
   Verifies the redirect does not break address computation.
2. **Multi-wave barrier**: two waves, wave 0 writes, `s_barrier`, wave 1 reads.
   Verifies the fence upgrade is correct (this is hazard #1).
3. **Reduction**: per-lane values summed via DS + barrier loop.
   Verifies multi-barrier correctness.

These are in `test-unit/HotswapMCTest.cpp` (runtime-executing) plus a lit test
`hotswap-lds-redirect-shape.s` that verifies the emitted IR shape without GPU.

### 9.2 IR shape lit test (`hotswap-lds-redirect-shape.s`)

Compiled from gfx1250 source; run through the transpiler with the redirect
forced.  FileCheck verifies:
- `wg_base` GEP chain (dispatch ptr loads → linear WG id multiply → GEP)
- DS reads produce `load` from `addrspace(1)`, not `addrspace(3)`
- DS writes produce `store` to `addrspace(1)`
- Each `llvm.amdgcn.s.barrier()` is bracketed by `fence release` (before) and
  `fence acquire` (after) with `syncscope("workgroup-one-as")`

### 9.3 Cross-lane exclusion test (`hotswap-lds-redirect-xlan.s`)

A kernel containing `ds_bpermute_b32`, `ds_swizzle_b32`, and plain
`ds_load_b32` / `ds_store_b32`.  With force-redirect on:
- `ds_bpermute` and `ds_swizzle` still lower to `addrspace(3)` intrinsics
- `ds_load`/`ds_store` redirect to `addrspace(1)` global

### 9.4 Capability demo — a kernel that does NOT fit (> 64 KB LDS)

Currently refuses; with `HSA_HOTSWAP_LDS_TO_GLOBAL=1` it runs on gfx1151 and
matches a CPU reference result.  Covered by the runtime A/B harness in
`rocm-hotswap-testing` (optional, requires live GPU).

---

## 10. Open questions / deferred design forks

1. **Kernarg extension protocol**: the current prototype adds a typed parameter
   at the end of the function signature.  This changes the AMDHSA `.args`
   metadata, which the HIP runtime uses for `kernelParams` scatter.  An
   alternative is injecting the pointer via a `hidden_` arg slot (invisible to
   the HIP layer) — lower risk but requires interceptor-side YAML patching.
   Evaluate once the runtime integration is wired.

2. **DS atomics via CAS loop**: `ds_add_u32` and friends can be emitted as
   `atomicrmw add ptr_global G1, monotonic, syncscope("workgroup-one-as")`.
   The AMDGPU backend will lower this to a global atomic instruction.  The open
   question is whether the global atomics have identical memory-model guarantees
   to the LDS atomics for the within-workgroup case.  Likely yes (both are
   sequentially consistent within the scope); requires verification.

3. **DS2 two-offset redirect**: straightforward extension — each DS2 op maps
   to two independent GEPs with correctly scaled offsets.  Identical to the
   existing `ds2IsRead` / `ds2IsWrite` handler body, with `addrspace(1)`
   instead of `addrspace(3)`.

4. **TR8/TR16 redirect**: the transpose read operations actually do LDS storage
   + cross-lane shuffle.  For the redirect case, we could replace them with a
   global load + `ds_bpermute` chain (already implemented for the non-redirect
   path), which would be correct since `ds_bpermute` doesn't allocate storage.
   Deferred because the corpus kernels that need >64 KB LDS don't yet include TR
   ops in the overflow region.

5. **Bounded pool (§7)**: prototype uses full-grid allocation.  The pool
   requires one global atomic at kernel entry/exit and knowledge of
   `num_resident_wgs`.  The resident-WG count can be queried via
   `hsa_agent_get_info(HSA_AGENT_INFO_COMPUTE_UNIT_COUNT)` × max-WG-per-CU.
   Implement once correctness is confirmed.
