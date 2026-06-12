# Qwen2.5-VL HotSwap Problem Kernel Manifest

This file records the exact Qwen2.5-VL kernels that were discovered during the
gfx1250 -> gfx950 HotSwap iteration. The companion file
`qwen2_5_vl_problem_kernels.allowlist` contains the full mangled kernel names,
one per line, suitable for `HSA_HOTSWAP_TRANSLATE_KERNELS=@...`.

The intent is to preserve the full-kernel inventory for a future integration
test suite. The existing `hotswap-raise/*.s` tests pin reduced lowering
properties; this manifest keeps the real kernels that motivated them.

## Groups

### Initial ATen Elementwise And Scalar Setup

These kernels appeared early in the Qwen rope/index path and are included in the
exact allowlist so future e2e tests can launch the same translated set:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 1-9
- `qwen2_5_vl_problem_kernels.allowlist`: lines 17-20
- `qwen2_5_vl_problem_kernels.allowlist`: lines 24-25

Future test shape: launch through the PyTorch/HIP path or extract the code
objects and verify they transpile as a batch with no trap stubs for listed
kernels.

### rocPRIM Nonzero / Reduce / Partition Path

These are the rocPRIM kernels from boolean indexing and nonzero-style
operations:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 10-13
- `qwen2_5_vl_problem_kernels.allowlist`: lines 15-16

Observed issues:

- `row_ror` DPP support was required before the bool reduce translated.
- Additional reduce variants were discovered by exact suffix (`EUlT_E4`,
  `EUlT_E5`, `EUlT_E6`) as larger masks were reached.
- A compact boolean-indexing probe still showed a count mismatch on one
  patterned large mask; this needs a future numeric integration test, not just a
  raiser/FileCheck test.

Future test shape: a runtime bool-index/nonzero harness that checks both count
and gathered values for representative masks, plus a raise-only fixture for the
specific DPP shapes. The reduced `row_ror` lowering is already pinned by
`c2_dpp_row_ror_refuse.s`.

### ATen Index Gather

Exact kernel:

- `qwen2_5_vl_problem_kernels.allowlist`: line 14

Observed issues:

- Initially faulted with illegal memory access on partial boolean masks.
- The root bug was scalar branch lowering for `s_cbranch_vcc*`: it used the
  current lane's VCC bit instead of full-wave VCC-zero/nonzero semantics.

Reduced test now present:

- `s_cbranch_vcc_wave_mask.s`

Future test shape: runtime boolean indexing harness for `x[mask]` with all-false,
all-true, partial, and larger patterned masks.

### Arange / Position-ID Elementwise Kernels

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 17-19
- `qwen2_5_vl_problem_kernels.allowlist`: lines 24-25

Observed issues:

- These were mostly exact trap-stub discoveries while moving through
  `get_vision_position_ids`.

Future test shape: focused PyTorch position-id construction or extracted-code
object launch for the same `arange` + add/mul sequence.

### CatArray / Torch Stack

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: line 21
- `qwen2_5_vl_problem_kernels.allowlist`: lines 26-27

Observed issues:

- `CatArrayBatchedCopy_contig<..., Li1, ...>` faulted in a compact
  `torch.stack([a, b, c], dim=0)` repro.
- Root cause was SMEM address lowering for `s_load ..., sREG offset:N
  scale_offset`: the SGPR offset was scaled, but the static immediate offset was
  dropped.
- `CatArrayBatchedCopy_contig<..., Li2, ...>` was later discovered in the Qwen
  path and should stay in the full-kernel inventory.
- `CatArrayBatchedCopy_contig<..., Li3, ...>` was discovered after fixing the
  index_put symbol-boundary over-scan and is now listed exactly as launched.

Reduced test now present:

- `smem_sgpr_imm_scale_offset.s`

Future test shape: runtime `torch.stack`/CatArray harness for multiple `Li*`
variants and tensor shapes.

### CompareFunctor Variants

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 22-23

Observed issues:

- The vectorized CompareFunctor was first hit by a compact compare repro and
  passed once listed.
- The unrolled CompareFunctor appeared in the Qwen stack/position path.

Future test shape: compact contiguous and non-contiguous `a < b` comparisons,
checking the dispatched variant and output correctness.

### Index Put / Scatter Assignment

Exact kernel:

- `qwen2_5_vl_problem_kernels.allowlist`: line 28

Observed issues:

- The exact `index_put_kernel_impl<OpaqueType<8>>` symbol does not contain the
  CAS sites. HotSwap was over-scanning beyond the selected ELF symbol into a
  later `cuda_take_put_kernel` symbol that did contain
  `global_atomic_cmpswap_b32`, then refusing the selected kernel for the later
  symbol's obstruction.
- The decoder now receives the selected symbol extent and stops before following
  kernels in the same `.text` section. A compact boolean-mask `index_put`
  runtime repro now passes for `n = 8, 32, 102`.

Reduced test now present:

- `c3_atomic_cas.s` pins the loud refusal for non-commutative atomic replica
  races.
- `qwen_index_put_symbol_boundary.s` pins the exact symbol-boundary regression:
  the selected kernel must raise even when a later symbol in the same object
  contains a CAS.

Future test shape: runtime `index_put`/scatter harness for the Qwen-style
assignment into `position_ids`, with duplicate-index and non-duplicate-index
cases.

### MaxNan Reduce

Exact kernel:

- `qwen2_5_vl_problem_kernels.allowlist`: line 29

Observed issues:

- Initially refused due to missing `v_pk_max_i16`, `v_pk_max3_i16`, and
  `v_max3_i16` support.
- Compact `torch.arange(n, dtype=torch.long).cuda().max()` passed for
  `n = 8, 128, 512` after the opcode support was added.
- Compact MaxNan reduction passed, but full Qwen later exposed the separate
  `index_put` CAS-loop blocker described above.

Reduced test now present:

- `v_pk_max_i16.s`

Future test shape: runtime long reduction/max harness plus a full extracted
MaxNan code-object transpile/launch test.

### Vision Tower Gather / BF16 Copy / Grid Math

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 30-32

Observed issues:

- Once the Qwen rope/position-id path completed, full prefill entered the
  vision tower and dispatched `vectorized_gather_kernel<16, long>` followed by a
  `bfloat16_copy_kernel_cuda` elementwise kernel. Both must be listed exactly to
  avoid trap stubs while the full model prefill path is tested.
- The vision cumulative-sequence path also dispatches an unrolled
  `BinaryFunctor<long, long, long, MulFunctor<long>>` for
  `grid_thw[:, 1] * grid_thw[:, 2]`.

Future test shape: compact `get_image_features`/vision-position fixture that
launches the gather and bf16 copy path without the full decoder.

### Vision Window Index Helpers

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 33-40

Observed issues:

- Full vision prefill later entered `get_vision_window_index`, which launches a
  long `compute_cuda_kernel`, a small-index select, an int direct-copy kernel,
  an int rocPRIM scan, and an int fill. These were trap stubs until listed.
- The same path then calls `torch.unique_consecutive`, adding a uint lookback
  scan-state init, a subalgo-8 int partition, and an unsigned-long-to-long
  transform.

Future test shape: compact `get_vision_window_index` fixture using the model's
`grid_thw` tensor.

### Vision Patch Embedding Conv3d

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 41-43

Observed issues:

- Full prefill reaches `self.visual.patch_embed.proj`, which dispatches a
  MIOpen/Tensile conv3d kernel named
  `Cijk_Ailk_Bljk_BBS_BH_MT128x32x16_..._WGM8`. Without listing it, the strict
  path builds trap stubs and MIOpen reports an internal error when launching the
  selected invoker.
- With the Tensile kernel listed, MIOpen proceeds to launch a
  `naive_conv_ab_nonpacked_fwd_ncdhw_ushort_double_ushort_0` fallback/helper and
  a float `BinaryFunctor<MulFunctor>` elementwise kernel.

Future test shape: compact patch-embedding conv3d fixture for this exact
MIOpen invoker.

### Vision Rotary Embeddings

Exact kernels:

- `qwen2_5_vl_problem_kernels.allowlist`: lines 44-72

Observed issues:

- After patch embedding, the vision forward path reaches
  `position_embeddings = (emb.cos(), emb.sin())`. The first strict trap-stub
  launch there was a CatArray stack variant for `OpaqueType<4>, Li2`, followed
  by a vectorized `cos_kernel_cuda` elementwise kernel.
- The next run reached the first visual transformer block and launched the
  companion `sin_kernel_cuda`, `bfloat16tofloat32_copy_kernel_cuda`, and
  `pow_tensor_scalar_kernel_impl<float, float>` kernels in the rotary/RMSNorm
  path.
- RMSNorm then dispatches a float `MeanOps` reduce, float in-place add, and
  vectorized `rsqrt_kernel_cuda`.
- After `rsqrt`, the same path dispatches float and BF16 `MulFunctor` kernels
  that apply the normalization scale.
- The following QKV linear setup dispatches a BF16 direct-copy kernel before the
  first hipBLASLt GEMM attempt.
- The first attention block then launches a second hipBLASLt GEMM kernel, a
  manual-unroll BF16-to-F32 copy, and a float neg kernel for `rotate_half`.
- The same attention path then uses a non-contiguous CatArray
  `OpaqueType<4>, Li3`, a vectorized float add, and an unrolled int add.
- SDPA attention next launches a vectorized float `AUnaryFunctor<MulFunctor>`.
- The SDPA core then launches a hipBLASLt `Cijk_Alik_Bljk_SB...` GEMM and a
  `softmax_warp_forward<float, float, float, 6, false, false>` kernel.
- SDPA's validity/masking checks then dispatch `isneginf_kernel_impl<float>` and
  a bool `and_kernel_cuda` reduction.
- The masking path then dispatches a float `FillFunctor` and a float
  `where_kernel_impl(bool, float, float)` kernel.
- The next exact hipBLASLt variant was
  `Cijk_Ailk_Bljk_SB_MT128x64x8_...`, distinct from the earlier
  `Cijk_Alik_Bljk_SB_MT128x64x12_...` kernel.
- The attention output reshape/projection path then dispatches a CatArray
  `OpaqueType<2>, Li4` contiguous copy variant.
- The missing-ISA HSA-tool path now uses `HSA_HOTSWAP_SOURCE_TARGET` and reaches
  a further hipBLASLt `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S...MT32x16x32...` GEMM.
- The local-label symbol-boundary fix makes that missing-ISA Tensile GEMM decode
  the full body instead of stopping at an internal local label.
- Target-created spills are now carried through the HIP/ROCclr launch ABI:
  ROCclr refreshes private/group segment sizes from the loaded HSA symbol, so
  rewritten kernels whose target KD requests scratch can launch with a nonzero
  AQL `private_segment_size` instead of inheriting the original gfx1250 zero.
- With spill support enabled, the formerly blocking `MT32x16x32` GEMM is
  dispatched repeatedly and the full-model run advances to ~4100 dispatches
  before a later unspecified launch failure near the vision attention
  `torch.split` path. This confirms the immediate blocker moved beyond the
  target-private-segment launch ABI mismatch.
- For gfx950 targets, `v_wmma_f32_16x16x32_bf16` now lowers through the direct
  K=32 `mfma_f32_16x16x32_bf16` intrinsic instead of the gfx942-style two-K16
  `_1k` chain. This removes an unnecessary decomposition boundary and matches the
  documented gfx950 Template-A path, but this large Tensile kernel still exceeds
  the strict target-private-segment budget, so more liveness/resource work is
  required.
- After that point, SDPA launches another exact missing-ISA hipBLASLt variant,
  `Cijk_Alik_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT32x32x16...`, which is now part
  of the allowlist inventory. It hits the same class of large-Tensile
  resource-pressure issue as the `MT32x16x32` variant.
- The same SDPA recovery/fallback sequence also dispatches the transposed
  `Cijk_Ailk_Bljk_S_B_Bias_HA_S_SAV_UserArgs_MT32x32x16...LBSPPA512...`
  variant. It is now listed so future runs do not execute its trap stub while
  investigating the real `TargetPrivateSegmentBudgetExceeded` blocker.

Future test shape: compact rotary-position embedding fixture around the vision
module's `emb.cos()` / `emb.sin()` path.

## Current Full-Kernel Status

The exact allowlist currently has 72 entries. The reduced raiser tests cover the
main implementation bugs found so far:

- `s_cbranch_vcc_wave_mask.s`
- `smem_sgpr_imm_scale_offset.s`
- `ttmp6_cluster_workgroup_id_init.s`
- `v_pk_max_i16.s`
- `c3_atomic_cas.s`
- `qwen_index_put_symbol_boundary.s`
- `qwen_tensile_local_label_boundary.s`
- existing `c2_dpp_row_ror_refuse.s`

The remaining work for a future suite is to run the full kernels, or compact
PyTorch/HIP launchers that dispatch them, and compare results on GPU. The current
manifest is the inventory input for that suite.
