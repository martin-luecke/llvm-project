# Hotswap: Architecture Overview

## What Hotswap is

Hotswap is a binary translator for AMD GPU code objects. It takes a compute
kernel compiled for one GPU instruction set and produces an equivalent code
object for another, so a kernel can run on hardware it was not built for.

Currently supported ISAs:

- Source ISA: gfx1250
- Target ISA: gfx942, gfx950

Hotswap is a compiler-side library, built into COMGR and exposed as
`amd_comgr_hotswap_transpile`. It runs at code-object load time: given a kernel
whose ISA does not match the device, it returns a translated code object to load
in its place.

## Lift to IR, then re-lower

Hotswap disassembles the source kernel, lifts each machine instruction to LLVM
IR, and re-lowers that IR through the in-tree AMDGPU backend for the target --
essentially a decompiler front end for machine code, with the recovered IR handed
straight back to `llc`.

Going through IR reuses the backend that already targets the destination:
instruction selection, register allocation, scheduling, and kernel-descriptor
emission all come for free. Hotswap only has to produce correct IR. Most of the
work is in the front end -- recovering faithful IR from machine code whose meaning
depends on the source hardware's execution model.

## The pipeline

The translation stages, end to end:

```
  code object (ELF / HSACO, source ISA)
        │
        │  extract .text, enumerate kernels, read each kernel's
        │  descriptor + argument metadata
        ▼
  ┌──────────────────────────────────────────────────────────┐
  │  per kernel:  raiseToIR   (source machine code -> IR)    │
  │     disassemble .text  (LLVM MCDisassembler)             │
  │     recover basic blocks from branch targets             │
  │     analyze indirect branches (computed PC)              │
  │     classify wave-size obstructions, choose wave mapping │
  │     check every EXEC writer is modeled                   │
  │     build one IR function, seed entry registers          │
  │     lift each instruction via a per-format handler       │
  │     promote register memory to SSA (mem2reg)             │
  │     apply cross-lane rewrites, final safety checks       │
  └──────────────────────────────────────────────────────────┘
        │  LLVM IR (one function per kernel)
        ▼
     opt -O2   ->   llc -mcpu=<target>   ->   llvm-mc   ->   ld.lld -shared
        │
        ▼
  code object (target ISA)  +  structured result / diagnostics
```

Orchestration is in [`pipeline.cpp`](../pipeline.cpp). The lift itself is
[`raiser.cpp`](../raiser.cpp). The pipeline shells out to `opt`, `llc`,
`llvm-mc`, and `ld.lld` as separate processes.

The recovered IR has a few defining traits:

- **One function per kernel**, using the AMDGPU kernel calling convention.
- **Registers are modeled as memory, then promoted.** Every architectural
  register (SGPRs, VGPRs, EXEC, VCC, SCC, M0, ...) gets an `alloca`. Handlers emit
  loads and stores against it, and `mem2reg` reconstructs SSA and phi nodes
  afterward.
- **Control flow** is recovered as a basic-block graph, with branch targets
  computed from instruction offsets. Loop and conditional structure re-emerges
  through the later LLVM passes. Indirect branches -- a computed program counter
  placed in a register and jumped to -- are resolved by a dedicated static
  analysis.
- **Instruction dispatch** uses LLVM's instruction tables: the disassembler
  produces `MCInst`s, each is mapped to an architecture-neutral opcode identity,
  and routed to a format handler by its TableGen instruction flags.

Running the full backend per kernel is expensive, so two properties keep the cost
down:

- **Lazy, per kernel.** A code object can carry hundreds of kernels of which a
  program calls only a few, so a kernel is translated the first time it is
  dispatched rather than all up front.
- **Cached.** A translated kernel is keyed on the source object and the target
  ISA and reused, so the cost is paid once and skipped on later loads.

## Overview: Three problems beyond opcode remapping

If the source and target agreed on the execution model, the instruction set, and
the interface a kernel presents to the runtime, this would be opcode remapping.
They don't, and three problems account for the complexity.

### 1. The execution model differs (wave size)

An AMD GPU executes a *wavefront* of work-items in lockstep under a bitmask
called EXEC, one bit per lane. gfx1250 wavefronts are 32 lanes wide. gfx942 and
gfx950 wavefronts are 64. The source machine code bakes the 32-lane assumption
into three observable places: the width of the EXEC mask, the semantics of every
cross-lane instruction (lane shuffles, permutes, ballots, lane-id counts), and
the bit patterns of lane-id and workgroup-rank arithmetic. Translating across
the width gap requires two things.

**Modeling the source execution mask explicitly.** The source uses EXEC to
switch individual lanes on and off around side effects -- stores, atomics,
cross-lane reads. IR has no wavefront concept, and one IR "thread" is one
work-item.
Hotswap therefore tracks the source EXEC mask as a value and guards each per-lane
side effect with an explicit "is this lane active?" branch keyed on the lane's
EXEC bit. Wave-uniform state (scalar registers) is written unconditionally.
Per-lane state (vector registers, memory) is written only under that guard. When
the mask is provably all-ones the guard folds away, so uniform code pays
nothing. In the source this mechanism is named *SIMT Predicated Execution* (SPE).

**Deciding what the extra lanes do.** Going from 32 source lanes to 64 target
lanes raises the question of how source semantics map onto the wider wave. That
mapping is a policy, called a *projection*, and it is chosen per kernel:

- The default for the 32->64 case treats the target wave as two independent
  source waves stacked together (lanes 0-31 and 32-63), each with its own modeled
  execution mask. The whole kernel body runs with hardware EXEC forced on so that
  wide, all-lanes-must-participate operations (notably matrix instructions) work,
  while per-lane side effects stay gated by the modeled mask. In the source this
  is `WaveNativeProjection`.
- An earlier policy, still used for same-width translation and for kernels
  launched narrower than one target wave, maps target lane *L* onto source lane
  *L mod 32*. In the source this is `ModuloReplicationProjection`.

A projection is only correct for kernels whose observable behavior does not
depend on the absolute lane count. A per-kernel analysis checks that and refuses
the kernels it cannot establish it for.

### 2. Some instructions have no target equivalent

Whole instruction classes exist on the source but not the target, and must be
re-expressed in terms the target has:

- **Matrix multiply.** gfx1250 has wave-32 matrix instructions (WMMA). The
  targets have wave-64 matrix instructions (MFMA) that distribute matrix
  elements across lanes differently. Hotswap lowers WMMA to MFMA by redistributing
  operands across lanes (via lane-permute reads), splitting the contraction
  dimension where needed, running the MFMA, and gathering the result back into the
  source layout.
- **Microscaled FP4 conversion.** A gfx1250 instruction dequantizes eight packed
  4-bit floats with a shared scale. On targets without it, Hotswap synthesizes the
  dequant as exact integer bit arithmetic on the float fields.
- **Tensor DMA.** gfx1250 has a hardware unit that moves tiled tensors between
  global memory and LDS. On targets without it, these are emulated by a
  build-time device-code runtime.
- Plus the smaller cases: scalar-float, dual-issue instruction pairs, split vs.
  combined wait counters, and the differing flat/scratch and buffer formats.

Each instruction is handled by a dedicated handler. Depending on the class, the
translation is exact, an approximation, or a refusal.

### 3. The kernel's boundary with the runtime

The runtime launches a kernel through a fixed interface: a set of scalar
registers preloaded with pointers and IDs, and a kernarg segment holding the
explicit arguments followed by a block of hidden arguments -- grid and block
sizes, remainders, and framework-specific entries such as printf or hostcall
buffers. Source and target disagree on this interface, and the same launch still
has to drive the translated kernel, so Hotswap reconstructs the interface instead
of translating it.

**Preloaded registers.** The kernel descriptor decides which scalar registers
hold the kernarg pointer, dispatch pointer, workgroup IDs, and the rest, and the
set and order shift across generations (e.g. gfx1250 adds argument preloading and a
wider count field), so the kernarg pointer is not reliably register 0. Hotswap
reads the descriptor, reconstructs what every register holds at entry, and seeds
those values in the IR so the backend reproduces them.

**Explicit arguments** keep their source byte offsets, so a kernarg buffer built
for the source kernel lands each argument where the translated kernel expects it.

**Hidden arguments.** The source reads these from the tail of its kernarg segment
or through an implicit-argument pointer, and different libraries use different
ones. The target kernel has no hidden-argument block, so Hotswap identifies each
field the source reads by its byte offset and computes the value from dispatch
state: block and grid sizes from the dispatch packet, block counts and remainders
from those, the queue pointer from its intrinsic. Fields it does not model, such
as the printf and hostcall buffers, are refused. A wrong offset would silently
produce a wrong value, so the classification has to be exact.

The output descriptor is regenerated by the backend from the lifted function.
Hotswap sets the function's attributes so the regenerated argument-segment, LDS,
and workgroup sizes match the source's, and refuses if the target lowering needs
more scratch memory than the source kernel used, because that would need adjustment
of the launch parameters, which is not supported at this time.

## Wave-size translation

The source runs 32 work-items per wavefront and the targets run 64. Bridging that
takes two independent decisions: how to represent the source's per-lane execution
mask in the IR, and how to map 32 source lanes onto a 64-lane target wave.

### Modeling the execution mask

A wavefront runs one instruction across all its lanes at once, and a bitmask
called EXEC determines which lanes are active. The source kernel writes EXEC
constantly -- a compare that switches lanes off, a mask saved and restored around
a divergent region -- so that stores, atomics, and cross-lane reads happen only
on the lanes meant to run them.

LLVM IR has no concept of wavefronts. Each IR thread is one work-item, with no way to
express "this lane is off for this instruction." The source's use of EXEC has to become
explicit in the IR to retain the semantics of the source on the target architecture.

Hotswap carries EXEC as an ordinary value (an alloca that mem2reg later promotes
to SSA). Each write then takes one of two paths, depending on what it targets:

- Wave-uniform writes -- scalar registers, and EXEC, VCC, and SCC themselves --
  happen unconditionally.
- Per-lane writes -- vector registers, memory, LDS, atomics -- are guarded. The
  lane computes its own index, tests its bit in the current EXEC value, and
  performs the write only if that bit is set.

The guard is a branch, `if (my EXEC bit is set) { ... }`. Because the condition
depends on the lane's identity, LLVM's divergence analysis treats the branch as
divergent and the AMDGPU backend lowers it back into hardware EXEC masking --
narrowing EXEC around the guarded write and restoring it afterward. So the
source's hardware EXEC becomes an explicit value in the IR and returns to hardware
EXEC in the output. Where the mask is provably all-ones, the common case of
straight-line code, the guard folds away and costs nothing.

This holds only if every EXEC write is one Hotswap models. Before translating, it
checks that every EXEC-writing instruction routes through this model and refuses
the kernel if one does not, so an unmodeled EXEC write can never quietly drop its
masking.

### Mapping the lanes

A 64-lane target wave has twice the lanes the source assumed, so Hotswap fills it
with two source waves side by side: lanes 0-31 run source wave 0, lanes 32-63 run
source wave 1, each with its own EXEC. At entry it forces the hardware EXEC to
all-ones and saves the real per-lane mask into the modeled EXEC value. The
hardware then runs all 64 lanes for the whole kernel, while the per-lane guards,
reading the modeled EXEC, still confine each side effect to the right lanes.
Running all 64 lanes is what lets the wide matrix instructions work, since they
need every lane, and the guards keep the memory writes correct.

Because the two source waves keep separate lane numbering and separate data, this
also covers kernels that run more than a single wave's work per target wave. The
one exception is a kernel launched with fewer work-items than a single target
wave. There Hotswap uses a simpler mapping,
*modulo-replication*: target lane L stands in for source lane L mod 32, so the
upper 32 lanes mirror the lower 32. That works only because those upper lanes have
no work of their own and stay inactive.

### When the mapping is safe

Packing is correct only when the two source waves neither depend on the wave's
width nor on each other's progress. Most compute meets both conditions: it reads
its inputs, does arithmetic, and writes its outputs, each wave's work independent
of the other's.

The width condition fails for anything that reads a lane's position or reaches
across lanes. A target lane's absolute position runs 0-63 where the source assumed
0-31, so such an operation has to be re-expressed against the 32 lanes of its own
source wave. Most have an equivalent form and are rewritten. A few, e.g. a rotate
across the full wave, have none and are refused.

The progress condition fails when one wave has to wait on the other, and there is
no rewrite for it. On the source the two waves are separate wavefronts the
hardware schedules independently, so one can run ahead while the other waits. 
Packed into a single wave they run in lockstep and lose that independence.
A kernel that relies on it, e.g. one wave producing while another waits on the result,
or specialized waves coordinating using separate barriers, has no faithful form
under packing. Ordinary barriers that all waves reach together still translate.
A dependence that needs one wave to advance while another is stalled does not.

Hotswap refuses many of the kernels it cannot translate, but this last class is
the edge of what the analysis covers: a cross-wave dependence it does not
recognise can slip through, and the packed kernel then deadlocks.

## Planned sections

- **From machine code to IR** -- decoding, the register model, and the
  per-instruction handlers.
- **ABI reconstruction** -- how the kernel descriptor and argument layout are read
  and rebuilt.
- **Matrix and low-precision lowering** -- WMMA to MFMA, tensor moves, and FP4
  conversion.
