//===- atomic-replica.h - scaled-dispatch atomic gating -------------------===//
//
// Shared policy for emitting atomic RMWs under a scaled dispatch, used by the
// FLAT/GLOBAL, MUBUF, and DS handlers. Under a scaled dispatch source lane `i`
// and its active replica `i+W_s` share one logical thread and both pass the
// `emitUnderExec` mask, so an atomic issues twice unless the site opts into the
// one-replica gate or refuses.
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_HOTSWAP_ATOMIC_REPLICA_H
#define COMGR_HOTSWAP_ATOMIC_REPLICA_H

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace COMGR::hotswap {

class RaiseContext;
struct DecodedInst;

// Decide how the atomic RMW `Di` must be emitted so it issues exactly once per
// source lane under a scaled dispatch. `Format` names the handler for the
// refusal diagnostic. Returns:
//
//   - Error: the form cannot be made replica-consistent and must be refused.
//     Every *returning* atomicrmw refuses: the lane and its replica each read a
//     different "old" value (the second issue sees the first's write) and there
//     is no replica-0 -> replica-1 broadcast to reconcile them.
//   - true: a store-only *non-idempotent* RMW (add/sub/fadd/xor/swap/...) that
//     would double-count; the caller must wrap its emit in
//     `emitAtomicUnderOneReplica`.
//   - false: no gating needed -- an idempotent RMW (and/or/min/max, where
//     re-applying the same operand is a no-op), or not a scaled dispatch at all
//     (WaveNative forces full-wave EXEC and plain / phantom-lane MODREP never
//     dispatches the replica lanes, so each atomic already issues once).
llvm::Expected<bool> needsOneReplicaGate(RaiseContext &Ctx,
                                         const DecodedInst &Di,
                                         llvm::StringRef Format);

// Emit `Emit` under `if (lane_id < W_s)` so exactly one of a source lane and
// its scaled-dispatch replica issues the atomic, matching native wave32.
void emitAtomicUnderOneReplica(RaiseContext &Ctx,
                               llvm::function_ref<void()> Emit);

} // namespace COMGR::hotswap

#endif // COMGR_HOTSWAP_ATOMIC_REPLICA_H
