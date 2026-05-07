//===- comgr-metadata.h - Metadata query internals ------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_METADATA_H
#define COMGR_METADATA_H

#include "comgr.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

namespace COMGR {
namespace metadata {

amd_comgr_status_t getMetadataRoot(DataObject *DataP, DataMeta *MetaP);

size_t getIsaCount();

const char *getIsaName(size_t Index);

amd_comgr_status_t getIsaMetadata(llvm::StringRef IsaName,
                                  llvm::msgpack::Document &MetaP);

bool isValidIsaName(llvm::StringRef IsaName);

amd_comgr_status_t getElfIsaName(DataObject *DataP, std::string &IsaName);

// `MemoryBufferRef`-friendly variant of `getElfIsaName` for callers that
// don't have a `DataObject` handle (e.g. the hotswap transpiler running
// over raw HSACO bytes from disk). Writes the canonical AMDGPU ISA string
// (`amdgcn-amd-amdhsa--<gfx>:sramecc±:xnack±`) into `IsaName`. Returns
// `AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT` on malformed ELF or unrecognised
// MACH value.
amd_comgr_status_t getElfIsaNameFromBuffer(llvm::MemoryBufferRef MB,
                                           std::string &IsaName);

// `MemoryBufferRef`-friendly metadata extractor that populates the
// caller-supplied `Doc` with the AMDGPU MsgPack metadata note found in
// `MB`. Walks both PT_NOTE program headers and SHT_NOTE sections, supports
// all four ELF endian/bit variants, and recognises `NT_AMD_HSA_METADATA`
// (older YAML, name="AMD"), `NT_AMDGPU_METADATA` (MsgPack, name="AMDGPU"),
// and the PAL metadata note (type 13). Sets `EmitIntegerBooleans` to
// `true` for the MsgPack-encoded formats and `false` for the YAML format
// — downstream `iterate_map_metadata` consumers use this to decide
// whether 0/1 integer values should be reported as booleans. Returns
// `true` if at least one recognised note was processed; `false` if none
// was found. Used by `getMetadataRoot` (DataObject-facing entry) and by
// the hotswap transpiler's `code_object_utils` (which has no DataObject).
llvm::Expected<bool> walkElfMetadataIntoDoc(llvm::MemoryBufferRef MB,
                                            llvm::msgpack::Document &Doc,
                                            bool &EmitIntegerBooleans);

amd_comgr_status_t lookUpCodeObject(DataObject *DataP,
                                    amd_comgr_code_object_info_t *QueryList,
                                    size_t QueryListsize);

amd_comgr_status_t getIsaIndex(const llvm::StringRef IsaName, size_t &Index);

bool isSupportedFeature(size_t IsaIndex, llvm::StringRef Feature);

} // namespace metadata
} // namespace COMGR

#endif
