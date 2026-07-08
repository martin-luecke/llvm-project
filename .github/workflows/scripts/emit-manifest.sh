#!/usr/bin/env bash
# emit-manifest.sh - record the exact versions under test into manifest.json.
#
# Captures the llvm-project PR commit (the transpiler source comgr is rebuilt
# from) plus the component commits baked into the runner image (the
# rocm-hotswap-testing harness, rocm-systems/ROCR runtime, the baked llvm-acc
# link tree), the ROCm version, and the image refs. Runs on the gfx950 build
# host (it introspects the runner image with a throwaway container).
#
# Env: PR_SHA PR_REF MODEL_IMAGE BUILD_LIT_IMAGE OUT
set -uo pipefail
: "${OUT:?}"; : "${MODEL_IMAGE:?}"
mkdir -p "$(dirname "$OUT")"

# One throwaway container read of all baked component commits + ROCm version.
COMP="$(docker run --rm "$MODEL_IMAGE" bash -lc '
  git config --global --add safe.directory "*" 2>/dev/null || true
  echo "HARNESS=$(git -C /workspace/rocm-hotswap-testing rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "ROCRSYS=$(git -C /workspace/rocm-systems rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "LLVMACC=$(git -C /workspace/llvm-acc rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "ROCMVER=$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
' 2>/dev/null || true)"

export M_HARNESS="$(printf '%s\n' "$COMP" | sed -n 's/^HARNESS=//p' | head -1)"
export M_ROCRSYS="$(printf '%s\n' "$COMP" | sed -n 's/^ROCRSYS=//p' | head -1)"
export M_LLVMACC="$(printf '%s\n' "$COMP" | sed -n 's/^LLVMACC=//p' | head -1)"
export M_ROCMVER="$(printf '%s\n' "$COMP" | sed -n 's/^ROCMVER=//p' | head -1)"
export M_IMG="$(docker inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "$MODEL_IMAGE" 2>/dev/null || true)"
export M_BL="$(docker inspect --format '{{if .RepoDigests}}{{index .RepoDigests 0}}{{end}}' "${BUILD_LIT_IMAGE:-}" 2>/dev/null || true)"

python3 - "$OUT" <<'PY'
import json, os, sys
def nz(x, d="unknown"): return x if x else d
json.dump({
    "llvm_project_pr": {"commit": nz(os.environ.get("PR_SHA")), "ref": nz(os.environ.get("PR_REF"))},
    "rocm_hotswap_testing": nz(os.environ.get("M_HARNESS")),
    "rocm_systems": nz(os.environ.get("M_ROCRSYS")),
    "llvm_acc_baked": nz(os.environ.get("M_LLVMACC")),
    "rocm_version": nz(os.environ.get("M_ROCMVER")),
    "runner_image": {"ref": nz(os.environ.get("MODEL_IMAGE")), "digest": os.environ.get("M_IMG", "")},
    "build_lit_image": {"ref": nz(os.environ.get("BUILD_LIT_IMAGE")), "digest": os.environ.get("M_BL", "")},
}, open(sys.argv[1], "w"), indent=2)
PY
echo "=== manifest.json ==="; cat "$OUT"
