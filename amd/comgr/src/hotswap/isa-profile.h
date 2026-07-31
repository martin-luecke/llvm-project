//===- isa-profile.h - Hotswap transpiler ---------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_ISA_PROFILE_H
#define HOTSWAP_TRANSPILER_ISA_PROFILE_H

#include "MCTargetDesc/AMDGPUMCTargetDesc.h" // AMDGPU::Feature* enum
#include "SIDefines.h"            // AMDGPU::UfmtGFX1{0,1}::UFMT_32_FLOAT
#include "Utils/AMDGPUBaseInfo.h" // AMDGPU::hasMAIInsts
#include "llvm/MC/MCSubtargetInfo.h"

namespace COMGR::hotswap {

// Snapshot of the capability bits the raiser actually branches on. Every
// field is derived directly from the MC subtarget feature bits that TableGen
// emits, so adding a new AMDGPU generation does not require touching this
// struct; we just read the already-defined FeatureFoo bit.
//
// This is a pure value snapshot -- the factory copies bits out of the
// MCSubtargetInfo and does not retain any reference to it. Callers must
// construct via `fromSubtarget`; there is intentionally no default ctor.
struct ISAProfile {
  unsigned WaveSize = 64;
  bool HasAgpr = false;
  bool HasMfma = false;
  bool HasVopd = false;
  bool HasScalarFp = false;
  // True iff the target can select `llvm.amdgcn.tanh.*` to native v_tanh_*
  // TRANS instructions. gfx1250/gfx13 have it; gfx942 does not, so
  // cross-target lifts must not emit the intrinsic there.
  bool HasTanhInsts = false;
  // True iff the subtarget exposes the gfx12-era WMMA instructions
  // (FeatureWMMA{128,256}bInsts). gfx11 WMMA is encoded via FeatureGFX11Insts
  // + VOP3P patterns and is not covered here; the only WMMA source we lift
  // today is gfx1250.
  bool HasWmmA12 = false;
  // True iff the subtarget exposes the gfx1250 TENSOR cnt unit
  // (FeatureGFX1250Insts gates the VIMAGE TENSOR pseudo-instructions
  // `tensor_load_to_lds_d{2,4}` and `tensor_store_from_lds_d{2,4}` --
  // see `isGFX125xOnly` in AMDGPU.td and the
  // `int_amdgcn_tensor_load_to_lds` /
  // `int_amdgcn_tensor_store_from_lds` intrinsics in
  // IntrinsicsAMDGPU.td:4213). The flag is consumed by `handleVIMAGE`
  // to discriminate between the same-target intrinsic-emit path and
  // the cross-target loud refusal: the gfx942 and earlier ISAs have
  // no equivalent hardware unit, so cross-target lifts must refuse.
  bool HasTensorOps = false;
  bool HasIeeeNumMinMaxAtomics = false;
  // True iff the subtarget exposes gfx950's MAI extensions on top of the
  // shared gfx9-family `MAIInsts` feature.  Distinct from `HasMfma`, which
  // is set on every gfx9-family target with MAI (gfx940 / gfx942 / gfx950).
  // The discriminator is needed for cross-target lowerings whose target
  // intrinsic exists only on gfx950 -- notably the scaled F8F6F4 MFMA
  // family (`int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`,
  // IntrinsicsAMDGPU.td:3694), which is the gfx950 cross-target for
  // `v_wmma_scale_f32_16x16x128_f8f6f4`.  gfx942 has `HasMfma == true` but
  // no scaled F8F6F4 hardware, so the WMMA-scale handler in
  // `handle-valu-vop3p.cpp` must gate on `HasGfx950Insts` rather than
  // `HasMfma` to avoid a silent miscompile (the `HasMfma` predicate would
  // emit `int_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` on gfx942, where the
  // intrinsic has no codegen pattern and llc would crash at lowering).
  bool HasGfx950Insts = false;
  // True iff the target backend can lower the generic FP8 conversion
  // intrinsics such as `int_amdgcn_cvt_pk_fp8_f32`. gfx942 and gfx950 both
  // expose this feature, so use it instead of the broader `HasMfma` whenever
  // a cross-target expansion depends specifically on FP8 conversion support.
  bool HasFP8ConversionInsts = false;
  // v_prng_b32 (FeaturePrngInst). Targets without it have no selection
  // pattern for llvm.amdgcn.prng.b32, so lifts must expand it in IR.
  bool HasPrngInst = false;
  // Target exposes the FP8/BF8 MFMA family (FeatureFP8Insts; gfx942 + gfx950).
  // Distinct from HasMfma, which is set on every gfx9 MAI target -- gfx90a /
  // gfx940 have MAI but no FP8 MFMA pseudos.
  bool HasFP8Insts = false;
  // gfx125 widens compute_pgm_rsrc2.USER_SGPR_COUNT from the older 5-bit
  // GFX6-GFX120 field to a 6-bit field. Keep this as an ABI property rather
  // than deriving it from a string at each use site.
  bool HasGfx125UserSgprCountField = false;
  // True iff the V# buffer-resource descriptor carries a 57-bit base and
  // 45-bit num_records field. SMEM buffer loads decode source descriptors via
  // this bit before rebuilding a target resource with LLVM's target lowering.
  bool Has45BitNumRecordsBufferResource = false;
  // Signed base-address payload width for V# buffer resources. gfx12+ exposes
  // 57 address bits; older targets use the 48-bit descriptor shape built by
  // LLVM's make.buffer.rsrc lowering. Cross-ISA descriptor rebuilds must prove
  // the decoded source base is representable in the target width.
  unsigned BufferResourceBaseBits = 48;
  // Word3 (the format/flags dword) of a raw buffer V# built for this target.
  // Both descriptor forms the MUBUF path emits store this dword verbatim --
  // the hand-built <4 x i32> resource, and operand 3 of
  // `llvm.amdgcn.make.buffer.rsrc` (SIISelLowering::lowerPointerAsRsrcIntrin
  // applies no target fix-up) -- so a descriptor rebuilt for a cross-ISA
  // target must carry the target's own encoding, not the source's.
  //
  // gfx10+ mirrors the high dword of SIInstrInfo::getDefaultRsrcDataFormat():
  // UFMT_32_FLOAT, RESOURCE_LEVEL=1, OOB_SELECT=3. OOB_SELECT is the
  // load-bearing field -- it selects raw byte-extent bounds. Leaving it at 0
  // (as the gfx9 encoding does) makes a stride-0 descriptor report zero
  // records, so every access is out of bounds: loads return 0, stores and
  // atomics are dropped.
  //
  // gfx9 keeps DATA_FORMAT=32 + NUM_FORMAT=FLOAT rather than
  // getDefaultRsrcDataFormat's DATA_FORMAT=8, because MI300 MUBUF checks
  // DATA_FORMAT != INVALID before committing a raw store; 32/FLOAT is the
  // bisected value and matches what native gfx942 Triton emits (see the
  // bisection table in mubuf-addr.cpp).
  uint32_t RawBufferRsrcWord3 = 0;
  // True iff the source ISA exposes 1024 addressable VGPRs
  // (FeatureGFX1250Insts / AMDGPU.td `1024-addressable-vgprs`). On these
  // targets s_setreg targeting HW_REG_MODE captures VGPR_MSB from the operand
  // bits [12:19]; on older targets those bits are ordinary FP-mode fields.
  bool Has1024AddressableVGPRs = false;
  // True iff the target supports the gfx11+ s_sendmsg(MSG_DEALLOC_VGPRS)
  // encoding; gfx942 and earlier reserve ID=3.
  bool SupportsDeallocVgprs = false;

  // Addressable (physical) LDS capacity of this target, in bytes
  // (IsaInfo::getAddressableLocalMemorySize). The async-to-LDS lowering uses it
  // as the out-of-range bound separating a real LDS destination from the gfx12
  // INT_MAX drop sentinel (see handle-flat.cpp).
  unsigned LdsByteCapacity = 65536;

  bool isWave32() const { return WaveSize == 32; }
  bool hasValidWaveSize() const { return WaveSize == 32 || WaveSize == 64; }

  static ISAProfile fromSubtarget(const llvm::MCSubtargetInfo &STI) {
    ISAProfile P;
    P.WaveSize = STI.hasFeature(llvm::AMDGPU::FeatureWavefrontSize32) ? 32 : 64;
    // AGPRs/MFMA share the mai-insts feature today; keep them as separate
    // fields so future divergence stays expressible without touching callers.
    P.HasMfma = llvm::AMDGPU::hasMAIInsts(STI);
    P.HasAgpr = P.HasMfma;
    P.HasVopd = STI.hasFeature(llvm::AMDGPU::FeatureVOPDInsts);
    P.HasScalarFp = STI.hasFeature(llvm::AMDGPU::FeatureSALUFloatInsts);
    P.HasTanhInsts = STI.hasFeature(llvm::AMDGPU::FeatureTanhInsts);
    P.HasWmmA12 = STI.hasFeature(llvm::AMDGPU::FeatureWMMA128bInsts) ||
                  STI.hasFeature(llvm::AMDGPU::FeatureWMMA256bInsts);
    P.HasTensorOps = STI.hasFeature(llvm::AMDGPU::FeatureGFX1250Insts);
    P.HasIeeeNumMinMaxAtomics = llvm::AMDGPU::isGFX12Plus(STI);
    P.HasGfx950Insts = STI.hasFeature(llvm::AMDGPU::FeatureGFX950Insts);
    P.HasFP8ConversionInsts =
        STI.hasFeature(llvm::AMDGPU::FeatureFP8ConversionInsts);
    P.HasPrngInst = STI.hasFeature(llvm::AMDGPU::FeaturePrngInst);
    P.HasFP8Insts = STI.hasFeature(llvm::AMDGPU::FeatureFP8Insts);
    P.HasGfx125UserSgprCountField = llvm::AMDGPU::isGFX1250Plus(STI);
    P.Has45BitNumRecordsBufferResource =
        STI.hasFeature(llvm::AMDGPU::Feature45BitNumRecordsBufferResource);
    P.BufferResourceBaseBits = P.Has45BitNumRecordsBufferResource ? 57 : 48;
    P.RawBufferRsrcWord3 = rawBufferRsrcWord3(STI);
    P.Has1024AddressableVGPRs =
        STI.hasFeature(llvm::AMDGPU::Feature1024AddressableVGPRs);
    P.LdsByteCapacity =
        llvm::AMDGPU::IsaInfo::getAddressableLocalMemorySize(&STI);
    P.SupportsDeallocVgprs = llvm::AMDGPU::isGFX11Plus(STI);
    return P;
  }

  // Test-only factory.  Constructs an `ISAProfile` with only the
  // `waveSize` dimension set (the other feature flags default to
  // `false`) so unit tests exercising wave-direction-gated code --
  // `WaveNativeProjection`'s ctor assertion, `emitLaneActiveBit`'s
  // source / target wave-width arithmetic, the
  // `providesFullWaveExecInvariant` contract --
  // don't have to stand up a full `MCSubtargetInfo` (which would
  // require pulling in the LLVM AMDGPU target init chain just to
  // read one bit).  Production code MUST use `fromSubtarget`:
  // hand-forging loses the cross-checks between feature flags
  // (e.g. `hasAGPR == hasMFMA`) that `fromSubtarget` derives from
  // the canonical subtarget feature definitions in LLVM's
  // AMDGPU.td.  The factory is named and scoped rather than a
  // public default ctor so `git grep forTesting` is the review
  // anchor, not `git grep 'ISAProfile()'` (which would also hit
  // the private default ctor declaration below and mask real
  // findings).
  static ISAProfile forTesting(unsigned WaveSize) {
    ISAProfile P;
    P.WaveSize = WaveSize;
    return P;
  }

private:
  // See `RawBufferRsrcWord3` above for why the target's own encoding is
  // required. gfx10+ reproduces the >= GFX10 branch of
  // SIInstrInfo::getDefaultRsrcDataFormat(); that helper needs a GCNSubtarget
  // (codegen) and cannot be called from the MC-only raiser, so the bit layout
  // is mirrored here against the same source.
  static uint32_t rawBufferRsrcWord3(const llvm::MCSubtargetInfo &STI) {
    if (llvm::AMDGPU::isGFX10Plus(STI)) {
      // getDefaultRsrcDataFormat() composes a 64-bit RSRC value; word3 is its
      // high dword. gfx10 and gfx11+ number UFMT_32_FLOAT differently.
      const uint64_t Ufmt32Float =
          llvm::AMDGPU::isGFX11Plus(STI)
              ? static_cast<uint64_t>(llvm::AMDGPU::UfmtGFX11::UFMT_32_FLOAT)
              : static_cast<uint64_t>(llvm::AMDGPU::UfmtGFX10::UFMT_32_FLOAT);
      constexpr uint64_t FormatShift = 44;
      constexpr uint64_t ResourceLevel = 1ULL << 56;
      constexpr uint64_t OobSelectRaw = 3ULL << 60;
      return static_cast<uint32_t>(
          ((Ufmt32Float << FormatShift) | ResourceLevel | OobSelectRaw) >> 32);
    }
    // gfx9 V# word3: NUM_FORMAT is bits [14:12], DATA_FORMAT bits [18:15].
    constexpr uint32_t DataFormat32 = 4u << 15;
    constexpr uint32_t NumFormatFloat = 7u << 12;
    return DataFormat32 | NumFormatFloat;
  }

  ISAProfile() = default; // constructible only via fromSubtarget() /
                          // forTesting(), per the comments above.
};

} // namespace COMGR::hotswap

#endif
