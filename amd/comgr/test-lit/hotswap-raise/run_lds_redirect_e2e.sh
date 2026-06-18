#!/usr/bin/env bash
# End-to-end smoke test for the LDS->global redirect path.
#
# Prerequisites:
#   - llvm-mc + ld.lld on PATH (or LLVM_BIN set)
#   - hotswap-transpile binary (or TRANSPILE set)
#   - hipcc on PATH
#   - gfx1151 GPU present
#
# Usage:
#   ./run_lds_redirect_e2e.sh [--build-dir <comgr-build>] [--llvm-bin <llvm/build/bin>]
#
# The script assembles lds_redirect_shape.s (gfx1250), translates it to
# gfx1151 with HSA_HOTSWAP_LDS_TO_GLOBAL=1 FORCE=1, compiles the HIP
# launcher, runs it, and verifies the 32-lane ring-shift result.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${BUILD_DIR:-}"
LLVM_BIN="${LLVM_BIN:-}"

# Parse args.
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --llvm-bin)  LLVM_BIN="$2";  shift 2 ;;
        *) echo "unknown arg $1"; exit 1 ;;
    esac
done

# Locate llvm-mc / ld.lld.
if [[ -z "$LLVM_BIN" ]]; then
    if command -v llvm-mc &>/dev/null; then
        LLVM_BIN="$(dirname "$(command -v llvm-mc)")"
    else
        echo "ERROR: llvm-mc not found; set LLVM_BIN or pass --llvm-bin" >&2
        exit 1
    fi
fi

# Locate hotswap-transpile.
TRANSPILE="${TRANSPILE:-}"
if [[ -z "$TRANSPILE" ]]; then
    if [[ -n "$BUILD_DIR" && -x "$BUILD_DIR/test-lit/hotswap-transpile" ]]; then
        TRANSPILE="$BUILD_DIR/test-lit/hotswap-transpile"
    elif command -v hotswap-transpile &>/dev/null; then
        TRANSPILE="$(command -v hotswap-transpile)"
    else
        echo "ERROR: hotswap-transpile not found; set TRANSPILE, BUILD_DIR, or PATH" >&2
        exit 1
    fi
fi

TMPDIR_E2E="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_E2E"' EXIT

echo "==> Assembling lds_redirect_shape_kernel (gfx1250)"
"$LLVM_BIN/llvm-mc" \
    -triple=amdgcn-amd-amdhsa -filetype=obj -mcpu=gfx1250 \
    "$SCRIPT_DIR/lds_redirect_shape.s" \
    -o "$TMPDIR_E2E/shape.o"
"$LLVM_BIN/ld.lld" -shared "$TMPDIR_E2E/shape.o" -o "$TMPDIR_E2E/shape_gfx1250.hsaco"

echo "==> Translating to gfx1151 (LDS->global redirect)"
HSA_HOTSWAP_LDS_TO_GLOBAL=1 HSA_HOTSWAP_LDS_TO_GLOBAL_FORCE=1 \
    "$TRANSPILE" \
    "$TMPDIR_E2E/shape_gfx1250.hsaco" \
    amdgcn-amd-amdhsa--gfx1250 amdgcn-amd-amdhsa--gfx1151 \
    --output="$TMPDIR_E2E/shape_gfx1151.hsaco"

echo "==> Compiling HIP launcher"
hipcc -std=c++17 \
    -o "$TMPDIR_E2E/lds_redirect_e2e" \
    "$SCRIPT_DIR/lds_redirect_e2e.cpp" \
    2>&1 | grep -v "^$" | grep -v "warning: ignoring return value" || true

echo "==> Running e2e test"
"$TMPDIR_E2E/lds_redirect_e2e" "$TMPDIR_E2E/shape_gfx1151.hsaco"
