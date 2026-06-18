# Design: gfx12 WMMA → gfx11 WMMA lowering (gfx1250 → gfx1151)

## Goal
Lift `v_wmma_f32_16x16x32_f16` (and `_bf16`) from a gfx1250 source so it runs on a
gfx11-family **wave32** target (gfx1151 / gfx1150) that has `v_wmma_f32_16x16x16_f16`
(K=16) but not the gfx1250 K=32 WMMA. This unblocks real matmul/attention kernels
(Triton + Tensile) for the gfx1151 target. f16 first; bf16 is the same path.

Out of scope for this target (gfx11 has no hardware): FP8/BF8 K=64, IU8, K=4 f32,
scaled F8F6F4 → these keep the existing principled refusal.

## Dispatch (handle-valu-vop3p.cpp, the K=32/K=64 WMMA case)
Follows the doc §5.0 native-vs-decompose pattern. New branch, inserted before the
final refusal:
```
if (Ctx.TargetIsa.HasTensorOps)            -> native gfx1250 K=32 WMMA   (existing)
else if (Ctx.TargetIsa.HasMfma)            -> emitWMMAtoMFMA              (existing)
else if (Ctx.TargetIsa.HasWmma16x16x16F16  -> emitWMMAtoGFX11WMMA         (NEW; F16/BF16 only)
         && (InputType==F16 || BF16))
else                                        -> unsupportedShape refusal   (existing)
```

## ISAProfile (isa-profile.h) — per-shape capability bit (doc §5.0.1)
Add, feature-derived (no target-triple string match):
```
bool HasWmma16x16x16F16 = false;  // target exposes int_amdgcn_wmma_f32_16x16x16_f16
...
P.HasWmma16x16x16F16 = AMDGPU::isGFX11Plus(STI) && P.isWave32();
```
gfx1250 also sets this, but HasTensorOps is checked first so it takes the native
K=32 path; gfx1151 (no HasTensorOps/HasMfma) reaches the new path. (BF16 reuses the
same bit; the 16x16x16 bf16 intrinsic is in the same gfx11 family.)

## emitWMMAtoGFX11WMMA (wmma-lowering.{h,cpp})
Wave32→wave32, so **no wave widening** (this is why it is much simpler than
emitWMMAtoMFMA): the projection is the same-wave identity (numSourceWavesPerTarget==1).
K=32 decomposes into 2× K=16 with accumulator chaining.

Fragment shapes (AMD Matrix Instruction Calculator / IntrinsicsAMDGPU.td):
- gfx12 source: A,B = `<16 x half>` (8 dwords/lane); C/D = `<8 x float>` (8 dwords/lane).
- gfx11 target: A,B = `<16 x half>`; C/D = `<8 x float>`. Intrinsic is 3-arg
  `int_amdgcn_wmma_f32_16x16x16_f16(A, B, C) -> D`.

Layout bridge (to be confirmed by the empirical probe + numeric test below):
- gfx12 A: lane `l` holds row `m=l%16`, K-elements `16*(l/16) .. +15` (lanes 0–15 → k0..15,
  lanes 16–31 → k16..31; no duplication).
- gfx11 A: lane `l` holds row `m=l%16`, k0..15, **duplicated** across lane halves
  (lanes 0–15 ≡ 16–31).
- So K-lo fragment `A1[l] = A[srcLane = l%16]`; K-hi `A2[l] = A[srcLane = (l%16)+16]`.
  Both are full-`<16 x half>` broadcasts → 8 `ds_bpermute` each (reuse
  unpackDwords/emitDSBpermute/packDwords). B identical.
- C/D: if gfx12 and gfx11 16×16 f32 accumulator layouts match (likely — same wave32
  16×16 tile), C passes straight in and D straight out, no bridge. The numeric test
  decides; add a C/D bpermute bridge only if it diverges.

Body:
```
A1=bcast(A, l%16);  B1=bcast(B, l%16);   D1 = wmma_16x16x16_f16(A1,B1,C)
A2=bcast(A,(l%16)+16); B2=bcast(B,(l%16)+16); D = wmma_16x16x16_f16(A2,B2,D1)
return D
```
Wrap bpermute/wmma outputs with `Ctx.Projection.wrapAsWWMValue` for partial-wave
safety (identity no-op under same-wave MODREP full-wave; matches emitWMMAtoMFMA).

## Verification (iterate until bit-exact)
1. **Layout probe** (rigorous): a native gfx1151 WMMA kernel that writes lane/VGPR →
   (m,n,k) tags, run on the laptop GPU, to confirm the gfx11 A/B/C/D equations.
2. **Numeric gate**: a minimal gfx1250 HIP WMMA 16×16×32 f16 single-tile matmul
   (`__builtin_amdgcn_wmma_*` or Triton's cached `mm_kernel.hsaco`), transpile
   gfx1250→gfx1151 with the CLI, run on the laptop GPU via a HIP loader, compare to a
   CPU/torch reference. Iterate the bridge (esp. C/D) until max-err ~0.
3. **lit test**: gfx1250 WMMA `.s` fixture → `raise_cli --target-isa=gfx1151 --emit-ir`,
   FileCheck for 2× `llvm.amdgcn.wmma.f32.16x16x16.f16` + the bpermute bridge +
   accumulator chaining; plus a negative test (FP8 K=64 still refuses on gfx1151).

## Build/iterate loop
`cmake --build build --target amd_comgr hotswap-transpile` (~1 min) → transpile test
kernel → run/compare → adjust. Keep changes additive and feature-gated so existing
gfx942/gfx950 MFMA paths are untouched.
