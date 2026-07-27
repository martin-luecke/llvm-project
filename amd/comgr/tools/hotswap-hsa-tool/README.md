# COMGR-backed HotSwap HSA API tool prototype

This directory contains the runtime-facing part of the cross-ISA HotSwap
prototype. It is a separate HSA API-tool DSO. Core `libamd_comgr` owns binary
translation and caching; this DSO owns presented-ISA API wrapping, lazy
dispatch orchestration, proxy queues, and runtime object lifetimes.

The prototype is deliberately opt-in and Linux-only. It is not linked into
`libamd_comgr`, and normal COMGR clients do not acquire an HSA runtime
dependency.

## Coordinated branches

The prototype requires the matching ROCm Systems branches:

- `users/mluecke/hotswap-api-tool-rocr` adds generic ROCr facilities for a
  physical execution-ISA query, explicit API-tool/profiler coexistence, and
  protected intercept-queue doorbells.
- `users/mluecke/hotswap-api-tool-clr` is stacked on that branch and separates
  CLR's presented/code-selection ISA from its physical execution ISA.

## Build

Configure COMGR as described in `amd/comgr/src/hotswap/README.md`, adding:

```text
-DCOMGR_BUILD_HOTSWAP_HSA_TOOL=ON
-DCOMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR=<rocm-systems>/projects/rocr-runtime/runtime/hsa-runtime
```

Then build the `comgr-hotswap-hsa-tool` target. The output DSO is named
`libhsa_hotswap_comgr.so`.

## Activation

Activation is explicit; the presentation variable alone does not load a DSO:

```sh
HSA_TOOLS_LIB=/path/to/libhsa_hotswap_comgr.so \
HSA_HOTSWAP_PRESENT_ISA=gfx1250 \
HSA_HOTSWAP_PROOF_LOG=/path/to/proof.jsonl \
your-program
```

`HSA_HOTSWAP_CACHE_DIR` selects a COMGR cache directory. The tool always asks
COMGR for strict per-kernel translation. For HIP workloads whose ABI contract
guarantees a zero global offset,
`HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO=1` passes that explicit assumption
to COMGR.

## Prototype limitations

- Only gfx1250 presentation on gfx942 or gfx950 has been exercised.
- The target-tagged skeleton object is a prototype mechanism for preserving
  source descriptors; it is not a finalized loader ABI.
- Independently translated kernels do not yet provide a general solution for
  writable program globals, arbitrary relocations, function pointers, indirect
  calls, or device enqueue. COMGR strict-mode refusals remain fatal.
- Translation is synchronous in the queue intercept callback. A production
  tool needs a specified asynchronous failure/queue-poisoning contract.
- Unsupported queue types and packet formats fail closed. The prototype does
  not claim cooperative, scheduler, or device-enqueue coverage.
- Mixed physical GPU processes and debugger/profiler attribution require more
  lifecycle and identity work.

The proof log records source-object registration, translation, cache status,
dispatch replacement, rejected objects, protected queues, and a final coverage
summary. It is validation evidence, not a security boundary.
