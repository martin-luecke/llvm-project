//===- comgr-env.h - Comgr environment variables --------------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_ENV_H
#define COMGR_ENV_H

#include "llvm/ADT/StringRef.h"

namespace COMGR {
namespace env {

/// Return whether the environment requests temps be saved.
bool shouldSaveTemps();
bool shouldSaveLLVMTemps();
std::optional<bool> shouldUseVFS();

/// If the environment requests logs be redirected, return the string identifier
/// of where to redirect. Otherwise return @p None.
std::optional<llvm::StringRef> getRedirectLogs();

/// Return whether the environment requests verbose logging.
bool shouldEmitVerboseLogs();

/// Return whether the environment requests time statistics collection.
bool needTimeStatistics();

/// If environment variable LLVM_PATH is set, return the environment variable,
/// otherwise return the default LLVM path.
llvm::StringRef getLLVMPath();

/// If environment variable AMD_COMGR_CACHE_POLICY is set, return the
/// environment variable, otherwise return empty
llvm::StringRef getCachePolicy();

/// If environment variable AMD_COMGR_CACHE_DIR is set, return the environment
/// variable, otherwise return the default path: On Linux it's typically
/// $HOME/.cache/comgr_cache (depends on XDG_CACHE_HOME)
llvm::StringRef getCacheDirectory();

/// If environment variable AMD_COMGR_DRIVER_OPTIONS_APPEND is set, return the
/// space-separated options to append to clang driver invocations.
llvm::StringRef getDriverOptionsAppend();

/// Return whether the environment requests the WaveNative projection be
/// disabled for wave32->wave64 cross-widening, forcing ModuloReplication
/// instead. Diagnostic ablation knob (HSA_HOTSWAP_DISABLE_WAVE_NATIVE); default
/// false so behavior is unchanged when unset.
bool shouldDisableWaveNative();

/// Return whether the environment requests the post-raise
/// cross-lane-divergent writelane/readlane rewrite be disabled. Diagnostic
/// ablation knob (HSA_HOTSWAP_DISABLE_WRITELANE_REWRITE); default false so
/// behavior is unchanged when unset.
bool shouldDisableWritelaneRewrite();

} // namespace env
} // namespace COMGR

#endif // COMGR_ENV_H
