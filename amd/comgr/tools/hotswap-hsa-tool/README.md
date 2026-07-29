# COMGR HotSwap HSA API tool

This directory builds `libhsa_hotswap_comgr.so`, an optional HSA API-tool DSO.
It owns the runtime-facing presentation and dispatch orchestration for COMGR's
HotSwap transpiler.  It is part of COMGR because its only translation boundary
is the public `amd_comgr_hotswap_transpile_with_options_v2` API and its release
lifecycle must match that API.  ROCr supplies generic API-tool loading,
intercept queues, and the physical execution-ISA query.  CLR consumes the
presented and execution ISA views but contains no translation policy.

Configure COMGR with `COMGR_ENABLE_HOTSWAP_TRANSPILE=ON` and
`COMGR_BUILD_HOTSWAP_HSA_TOOL=ON`.  The latter also requires
`COMGR_HOTSWAP_HSA_RUNTIME_INCLUDE_DIR` to name a current ROCr
`runtime/hsa-runtime` source directory.

## Activation

The tool is loaded only through `HSA_TOOLS_LIB`:

```text
HSA_TOOLS_LIB=/path/to/libhsa_hotswap_comgr.so
HSA_HOTSWAP_PRESENT_ISA=amdgcn-amd-amdhsa--gfx1250
```

`HSA_HOTSWAP_PRESENT_ISA` may also be the processor name alone, for example
`gfx1250`.  Loading the DSO without that variable is deliberately inert: no API
table is changed.  Setting the variable without loading the DSO has no effect.

The physical translation target is always obtained from
`HSA_AMD_AGENT_INFO_EXECUTION_ISA`.  It has no environment override and is
never replaced by the presented ISA.  Initialization fails when ROCr does not
provide the query.

Optional variables have these narrowly defined uses:

- `HSA_HOTSWAP_CACHE_DIR` selects COMGR's translation-cache directory.
- `HSA_HOTSWAP_PROOF_LOG` appends JSON-lines registration, translation,
  dispatch, and unload coverage events.  Every event carries the process ID so
  logs shared by compiler workers remain auditable.  Records escape every JSON
  control byte and are appended under an inter-process file lock, including
  partial-write retries.  A configured log that cannot be written is fatal.
- `HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO=1` passes COMGR's explicit strict
  global-offset assumption.  Without it COMGR retains its refusal behavior.

The preload-based legacy dispatch interceptor is neither loaded nor used.

## Presented view and executable model

For translated GPU agents the tool wraps agent ISA iteration and the agent
`ISA`, `NAME`, and `WAVEFRONT_SIZE` queries.  The presented ISA is an ISA handle
registered by ROCr, so ROCr remains authoritative for ISA names, target
features, compatibility, exception policies, rounding modes, and wavefront
properties.  All other agent queries, especially
`HSA_AMD_AGENT_INFO_EXECUTION_ISA`, pass through.

Source ELF bytes are captured by the memory, file, and loader file-slice
code-object-reader APIs.  COMGR's public metadata APIs validate the source ISA
and discover kernel names and ABI sizes.  COMGR's symbol API and LLVM's ELF
object model verify that the source contains no program storage.  The only
object symbols admitted are defined, global, metadata-matched kernel
descriptors and Clang's one-byte HIP compilation-unit marker; writable
sections must likewise be loader-owned or fully accounted for by those
markers.  This restriction is required because independent per-kernel child
executables cannot preserve one shared device-global identity.  Source machine
code is never passed to ROCr's loader.  A source object is registered against
its otherwise empty parent executable.  Lookup and iteration return opaque
virtual symbol handles whose source-facing attributes are synthesized from
the validated metadata.  Their kernel objects are separate non-executable
tokens.  The first protected dispatch of a token translates that kernel,
loads one physical child executable, and substitutes the translated kernel
object.  Lookup, iteration, executable validation, and ordinary symbol
attributes do not translate unrelated kernels.

A virtual kernel token is not a loader-managed device address.  Before its
first dispatch, the loader host-address query therefore returns
`HSA_STATUS_ERROR_INVALID_ARGUMENT` with a null output instead of fabricating a
descriptor or translating the kernel as a side effect of introspection.  This
also lets clients conservatively disable descriptor-based preload
optimizations.  After translation, the query is forwarded for the physical
child's kernel object.

This replaces the prototype's target-tagged source skeleton.  Changing only an
ELF target tag would leave incompatible instructions and generation-specific
descriptors reachable by the physical loader and is not a valid executable
contract.

The following core executable operations are wrapped to maintain this model:

- reader creation and destruction, including loader file slices;
- agent/program/legacy code-object loading and variable definition;
- freeze, validation, and destruction;
- name-based and deprecated symbol lookup;
- symbol information and all symbol-iteration forms.

Program code objects, client-visible program/agent globals, legacy code-object
loading for a presented agent, mixed native/source content in one executable,
dynamic-callstack kernels, module-linkage kernel descriptors, and more than one
source object in a parent executable are unsupported.  They return an HSA error
rather than loading source code.  Loader segment,
loaded-code-object, and executable-wide introspection returns
`HSA_STATUS_ERROR_NOT_SUPPORTED` while virtual executables exist because the
public loader API cannot describe the parent/child virtualization without
exposing physical children.

## Queues and packets

Core `hsa_queue_create`, `hsa_soft_queue_create`,
`hsa_amd_queue_intercept_create`, and the AMD batch queue-create API are
wrapped.  A translated agent accepts only interceptable, host-visible,
multi-producer compute queues.  Soft queues have no agent identity and expose
an application-owned doorbell, so they are refused whenever any agent needs
translation.  AMD batch descriptors retain priority and CU-mask settings;
version-1's unspecified group-segment size uses ROCr's documented
`UINT32_MAX` default.  Single-producer/device, SDMA, AIE, cooperative,
non-zero placement-flag, malformed, and otherwise unsupported queue forms are
rejected.

The interception callback is bound to the returned queue's actual packet
capacity and rejects a larger batch before copying packet bytes.  The tool
supports ordinary kernel-dispatch packets, barrier-and/or packets,
AMD barrier-value packets, and the non-clustered extended kernel dispatch that
can be losslessly lowered to an ordinary dispatch.  Extended-dispatch
dependency signals and performance hints cannot be represented by an ordinary
dispatch and are therefore refused rather than discarded.  Agent dispatch,
clustered dispatch, scheduler packets, unknown packet types, and unknown
vendor formats are fatal before submission.  Intercept-marker packets are
also refused: their callback receives ROCr's wrapped hardware queue and is
therefore not a safe interface for a protected queue.

Each kernel dispatch must contain a registered virtual token.  The rewrite
atomically validates and applies:

```text
packet segment size - source fixed size + target fixed size
```

as well as any COMGR-provided x-dimension projection scale.  Workgroup and grid
dimension and total-size limits are queried from the physical execution ISA.
Underflow, overflow, a scaled dispatch beyond those physical limits, a missing
target, an unregistered object, or translation failure aborts before the
packet writer can reach the hardware queue.  ROCr must also refuse the
doorbell-ID query on intercept queues; that is a generic intercept queue
invariant, not HotSwap policy.

Translated child executables live until the parent executable is destroyed;
their code-object readers are released as soon as loading completes.  Per-
kernel translation is serialized and cached in the record, so concurrent and
repeated dispatches issue one COMGR request.  HSA's normal rule that an
executable must not be destroyed while its kernels are in flight also governs
the virtual parent.  Symbol iteration takes an ownership lease rather than
holding a tool lock across an application callback.  Parent destruction marks
the virtual executable unusable immediately, while the final active iteration
releases its child and symbol mappings after the callback returns.  This also
permits a callback to re-enter executable destruction without deadlock.
Retired virtual symbol and kernel-token allocations remain reserved until tool
unload, so an allocator cannot reuse a stale opaque handle or kernel-object
address for an unrelated executable later in the process.

Every protected queue must be destroyed before the final `hsa_shut_down`.
ROCr invokes tool unload before it releases agents and asynchronous signal
handlers, so retaining a callback into an unloaded tool would be unsafe.  The
tool stops teardown with a diagnostic if a protected queue is still live.
Applications that follow the HSA queue-destruction lifecycle are unaffected.

API tables are restored on unload, and a coverage summary requires every
intercepted dispatch to have been rewritten.  ROCr calls tool unload after its
public HSA reference count reaches zero, so public executable-destruction calls
are no longer valid there.  The tool releases its remaining host records and
ROCr's still-live loader destroys any untranslated parent and translated child
during the immediately following runtime teardown.  A slot still owned
directly by this tool is restored to its saved entry.  A slot wrapped later by
another tool is left untouched rather than clobbering that outer wrapper; ROCr
performs its normal full API-table reset immediately after reverse-order tool
unload completes.
