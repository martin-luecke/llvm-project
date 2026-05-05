//===- handlers.h - Hotswap transpiler ------------------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef HOTSWAP_TRANSPILER_HANDLERS_H
#define HOTSWAP_TRANSPILER_HANDLERS_H

#include "raise_context.h"

namespace llvm {
class MCInstrInfo;
} // namespace llvm

namespace COMGR::hotswap {

class OpcodeMap;

// Asserts every MFMA-format opcode the disassembler can decode has a CanonicalOp
// handler entry. See `handle_mfma.cpp` for details.
void verifyMFMACoverage(const llvm::MCInstrInfo &MCII, const OpcodeMap &opcMap);

HandlerResult handleSOPP(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSMEM(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOPC(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOP1(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOPK(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleSOP2(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleVALU(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleFLAT(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleDS(RaiseContext &ctx, const DecodedInst &di,
                       OpResolver &op);
HandlerResult handleMUBUF(RaiseContext &ctx, const DecodedInst &di,
                          OpResolver &op);
HandlerResult handleMFMA(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleVOPD(RaiseContext &ctx, const DecodedInst &di,
                         OpResolver &op);
HandlerResult handleVIMAGE(RaiseContext &ctx, const DecodedInst &di,
                           OpResolver &op);

} // namespace COMGR::hotswap

#endif
