//===- transpile.cpp - COMGR hotswap transpile wrapper -------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "transpile.h"

#include <amd_comgr.h>
#include <cstdio>
#include <cstdlib>

namespace hotswap {
namespace interposer {

int retargetCodeObject(const void *ElfData, size_t ElfSize,
                       const char *SourceIsa, const char *TargetIsa,
                       void **OutData, size_t *OutSize) {
  if (!OutData || !OutSize)
    return -1;
  *OutData = nullptr;
  *OutSize = 0;
  if (!ElfData || ElfSize == 0 || !SourceIsa || !TargetIsa)
    return -1;

  amd_comgr_data_t Input = {0};
  amd_comgr_status_t St =
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &Input);
  if (St != AMD_COMGR_STATUS_SUCCESS)
    return static_cast<int>(St);

  St = amd_comgr_set_data(Input, ElfSize, static_cast<const char *>(ElfData));
  if (St != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(Input);
    return static_cast<int>(St);
  }

  amd_comgr_data_t Output = {0};
  St = amd_comgr_hotswap_transpile(Input, SourceIsa, TargetIsa, &Output);
  amd_comgr_release_data(Input);
  if (St != AMD_COMGR_STATUS_SUCCESS) {
    std::fprintf(
        stderr,
        "[hotswap-interposer] COMGR transpile %s -> %s failed (rc=%d)\n",
        SourceIsa, TargetIsa, static_cast<int>(St));
    return static_cast<int>(St);
  }

  size_t Size = 0;
  St = amd_comgr_get_data(Output, &Size, nullptr);
  if (St != AMD_COMGR_STATUS_SUCCESS || Size == 0) {
    amd_comgr_release_data(Output);
    return St != AMD_COMGR_STATUS_SUCCESS ? static_cast<int>(St) : -1;
  }

  void *Buf = std::malloc(Size);
  if (!Buf) {
    amd_comgr_release_data(Output);
    return -1;
  }
  St = amd_comgr_get_data(Output, &Size, static_cast<char *>(Buf));
  amd_comgr_release_data(Output);
  if (St != AMD_COMGR_STATUS_SUCCESS) {
    std::free(Buf);
    return static_cast<int>(St);
  }

  *OutData = Buf;
  *OutSize = Size;
  return 0;
}

} // namespace interposer
} // namespace hotswap
