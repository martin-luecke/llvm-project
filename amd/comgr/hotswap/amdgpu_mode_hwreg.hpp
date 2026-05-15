//===-- amdgpu_mode_hwreg.hpp - SQ wave MODE register helpers -------------===//
//
// Named bit positions and decode helpers for the per-wave MODE register
// (HW_REG_MODE / HW_REG_WAVE_MODE on gfx12+). Field layout matches
// s_setreg/s_getreg simm16 encoding:
//   id[5:0] | offset[10:6] | (size-1)[15:11]
//
// Canonical gfx1250 kernel prologue (before the first SMEM/VMEM op):
//   s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, ModeReg::ReplayModeBit, 1), 1
//
//===----------------------------------------------------------------------===//

#ifndef AMD_COMGR_HOTSWAP_AMDGPU_MODE_HWREG_HPP
#define AMD_COMGR_HOTSWAP_AMDGPU_MODE_HWREG_HPP

#include <cstdint>

namespace transpiler {
namespace amdgpu {

/// HWREG id for the wave MODE register (`HW_REG_MODE` / `HW_REG_WAVE_MODE`).
static constexpr unsigned HwregIdMode = 1;

/// Per-wave MODE register bit fields.
struct ModeReg {
  /// FP16_OVFL — overflowed f16 VALU results clamp to +/-MAX_FP16 instead of
  /// +/-inf (true infinities are preserved).
  static constexpr unsigned Fp16OvflBit = 23;

  /// REPLAY_MODE — 0 = single-VMEM-group replay (hardware XCNT waits); 1 =
  /// multi-VMEM-group replay (software inserts s_wait_xcnt). Must be programmed
  /// before the first SMEM/VMEM instruction on the wave.
  static constexpr unsigned ReplayModeBit = 25;
  static constexpr unsigned ReplayModeFieldSizeBits = 1;
  static constexpr unsigned ReplayModeMultiGroup = 1;
};

/// Decoded s_setreg / s_getreg field selector from the simm16 immediate.
struct SetregField {
  unsigned hwregId;
  unsigned offset;
  unsigned sizeBits;
};

inline SetregField decodeSetregSimm16(int64_t simm16) {
  const uint32_t enc = static_cast<uint32_t>(simm16) & 0xffffu;
  return {enc & 0x3fu, (enc >> 6) & 0x1fu, ((enc >> 11) & 0x1fu) + 1u};
}

/// True when \p simm16 / \p imm encode
/// `s_setreg_imm32_b32 hwreg(HW_REG_WAVE_MODE, ModeReg::ReplayModeBit, 1), 1`.
inline bool isModeReplayMultiGroupWrite(unsigned hwregId, int64_t simm16,
                                        int64_t imm) {
  const SetregField field = decodeSetregSimm16(simm16);
  return hwregId == HwregIdMode &&
         field.offset == ModeReg::ReplayModeBit &&
         field.sizeBits == ModeReg::ReplayModeFieldSizeBits &&
         imm == ModeReg::ReplayModeMultiGroup;
}

} // namespace amdgpu
} // namespace transpiler

#endif // AMD_COMGR_HOTSWAP_AMDGPU_MODE_HWREG_HPP
