# Converging with native runtime HotSwap

ROCm/rocm-systems PR #7921 ("feat(rocr): integrate HotSwap into ROCR loader",
ROCM-27304) moves HotSwap out of the `projects/hotswap` `HSA_TOOLS_LIB` plugin
and into the ROCR loader itself. This note records how this interposer is shaped
to converge with that direction rather than fight it. (The PR is fresh and will
change; we deliberately do not build on its internals — we match its conventions.)

## What the native integration does

- Intercepts at `hsa_executable_load_agent_code_object` (in
  `core/runtime/hsa.cpp`) via `hotswap::LoadAgentCodeObjectWithHotswap` -- the
  same load seam this interposer's tool half hooks through the `CoreApiTable`.
- Resolves the rewrite target from the **agent's** ISA name
  (`hsa_isa_get_info_alt`) plus `HSA_AMD_AGENT_INFO_ASIC_REVISION`, and currently
  gates on `gfx_target == "gfx1250" && asic_revision == 0` -- i.e. the gfx1250
  **B0 -> A0** stepping case, via the byte-patch `amd_comgr_hotswap_rewrite`.
  Cross-family (gfx1250 -> gfx1151/942/950) is not handled there.
- Lazy-loads COMGR (preferring one beside `libhsa-runtime64.so`).
- Keeps `HSA_HOTSWAP_DISABLE` and `HSA_HOTSWAP_VERBOSE`.

## The collision, and how we avoid baking it in

Native target resolution reads the **agent ISA**. Under our spoof the agent ISA
is the spoofed source (gfx1250), so an agent-ISA-derived target can never name
the real device -- the only component that knows the real device beneath the
spoof is this interposer. So the convergence interface is a **target override**:

- The LD_PRELOAD (spoof) half detects the real device before installing the
  spoof and publishes it as **`HSA_HOTSWAP_TARGET`** (`setenv`, without
  clobbering a user-set value).
- The transpile decision -- here, and intended for the native loader -- is
  simply: **transpile iff the captured code object's gfx differs from
  `HSA_HOTSWAP_TARGET`; otherwise forward.** Same shape as the native
  "rewrite on mismatch", with the target supplied explicitly instead of derived
  from the (spoofed) agent.

This keeps the two halves agreeing on the real ISA and means the natural upstream
change is small and obvious: let the native loader honour an explicit
`HSA_HOTSWAP_TARGET` and use the cross-family transpile entry point
(`amd_comgr_hotswap_transpile_with_options`) when source gfx != target gfx.

## Shared env conventions (this interposer honours all of these)

- `HSA_HOTSWAP_TARGET` -- real device ISA to transpile toward; auto-published by
  the spoof half, user-overridable.
- `HSA_HOTSWAP_DISABLE` -- forward every code object untouched (no transpile).
- `HSA_HOTSWAP_VERBOSE` -- diagnostics (also `HOTSWAP_INTERPOSER_LOG`).
- `HOTSWAP_INTERPOSER_SPOOF` -- interposer-only: the gfx target to present to the
  stack. This is the device-spoof mechanism, which the native integration does
  not have and which remains this project's distinct contribution.

## Division of labour at convergence

- **Native runtime HotSwap** (PR #7921, extended): the single capture+transpile
  engine at the load seam -- both A0/B0 rewrite and cross-family transpile,
  honouring `HSA_HOTSWAP_TARGET`.
- **This interposer**: the **device spoof** (so the stack emits the source ISA
  and the CLR fatbin-selection / ISA-compat source patches are unnecessary) and
  the **doorbell completeness backstop**. Neither is in PR #7921.

Until the native loader honours `HSA_HOTSWAP_TARGET` cross-family, the interposer
runs its own tool half. On a ROCr that already hotswaps natively at the agent
load path, set `HSA_HOTSWAP_DISABLE=1` to avoid double-processing on that path
(the interposer's program-scope / deprecated-path coverage, spoof, and backstop
remain additive).
