#!/usr/bin/env bash
# Run one HotSwap PyTorch-path model e2e compare (local gfx950 vs hotswap
# gfx1250->target) inside the PR#10 image, then evaluate the gate. Used by the Pytorch lane of hotswap-pr-ci.yml. Runs on the self-hosted runner HOST.
#
# Args:   $1 NAME     leg name (= config stem, e.g. whisper_small)
#         $2 SUBPATH  model dir under $WEIGHTS_HOST (e.g. openai/whisper-small)
# Env:    IMAGE         PR#10 image with the pytorch hotswap stack
#         WEIGHTS_HOST  host dir mounted at /data (model is /data/$SUBPATH)
#         SCRATCH_BASE  host dir for per-model scratch
#         GPU           GPU index (e.g. 7)
#         GITHUB_STEP_SUMMARY, GITHUB_WORKSPACE  (from Actions)
#
# Exit: 0 = gate PASS or weights-absent (skipped); 1 = gate FAIL or run error.
# NOTE: invoke with a shell WITHOUT errexit (find over the root-owned scratch
# dir exits non-zero on permission-denied subpaths).
set -uo pipefail

NAME="$1"; SUBPATH="$2"
HOSTPATH="$WEIGHTS_HOST/$SUBPATH"
SCRATCH="$SCRATCH_BASE/$NAME"
GATE="$GITHUB_WORKSPACE/.github/workflows/scripts/gate-pytorch.py"
# Lane label used by the consolidated summary aggregator (passed by workflow).
LANE="${LANE:-Pytorch E2E gfx950}"
RESULT_JSON="${RESULT_JSON:-$RUNNER_TEMP/model-result.json}"

# Write the machine-readable per-model result consumed by render-summary.py.
# Args: $1 state (pass|diverged|fail|skip)  $2 detail string.
write_result() {
  LANE="$LANE" MODEL="$NAME" STATE="$1" DETAIL="$2" \
    python3 - "$RESULT_JSON" <<'PY'
import json, os, sys
json.dump({
    "lane":   os.environ["LANE"],
    "model":  os.environ["MODEL"],
    "state":  os.environ["STATE"],
    "detail": os.environ["DETAIL"],
}, open(sys.argv[1], "w"))
PY
}

if [ ! -e "$HOSTPATH" ]; then
  write_result skip "weights not present at $HOSTPATH"
  echo ":fast_forward: \`$NAME\` ($LANE): SKIPPED — weights absent." >> "$GITHUB_STEP_SUMMARY"
  echo "SKIPPED (no weights: $HOSTPATH)"
  exit 0
fi

# Prior run scratch is root-owned (the model container runs as root); a host-side
# rm would hit "Permission denied". Clean it in a throwaway root container.
mkdir -p "$SCRATCH_BASE"
docker run --rm -v "$SCRATCH_BASE":/sb "$IMAGE" rm -rf "/sb/$NAME" 2>/dev/null || true
mkdir -p "$SCRATCH"
# Optional: mount PR-built comgr over the image comgr (set by the gated
# pipeline via download-artifact). Unset/missing -> use image-baked comgr.
PR_COMGR_MNT=()
if [ -n "${PR_COMGR:-}" ] && [ -f "$PR_COMGR" ]; then
  echo "Using PR-built comgr: $PR_COMGR"
  PR_COMGR_MNT=(-v "$PR_COMGR":/workspace/llvm-acc/build/lib/libamd_comgr.so.3.3.0:ro)
else
  echo "PR_COMGR not provided -- using image-baked comgr"
fi
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add 44 --group-add 109 \
  --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --network=host \
  -v "$WEIGHTS_HOST":/data:ro \
  -v "$SCRATCH":/scratch \
  "${PR_COMGR_MNT[@]}" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    cd /workspace/rocm-hotswap-testing
    export LD_LIBRARY_PATH=/workspace/llvm-acc/build/lib:$LD_LIBRARY_PATH
    make pytorch-hotswap-model-e2e \
      PYTORCH_MODEL_CONFIG=data/pytorch/models/'"$NAME"'.json \
      PYTORCH_MODEL_PATH=/data/'"$SUBPATH"' \
      PYTORCH_PYTHON=/workspace/venv/bin/python \
      PYTORCH_LIBHSA_RUNTIME=/workspace/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so \
      PYTORCH_INTERCEPT_LIB=/workspace/rocm-hotswap-testing/build/libhotswap_intercept.so \
      PYTORCH_GFX_OVERRIDE_LIB=/workspace/rocm-hotswap-testing/build/libhip_device_gfx_override.so \
      PYTORCH_COMGR_RUNTIME=/workspace/llvm-acc/build/lib/libamd_comgr.so.3.3.0 \
      PYTORCH_ROCBLAS_TENSILE_LIBPATH=/workspace/rocm-hotswap-testing/data/kernels/gfx1250_b0/tensile \
      PYTORCH_SCRATCH_ROOT=/scratch \
      PYTORCH_GPU='"$GPU"'
  ' 2>&1 | tee "$SCRATCH/run.log"
docker_rc=${PIPESTATUS[0]}
# Hand scratch back to the runner user (container wrote it as root) so this
# script AND the runner's _work/_temp cleanup can delete it next time.
docker run --rm -v "$SCRATCH_BASE":/sb "$IMAGE" chown -R "$(id -u):$(id -g)" "/sb/$NAME" 2>/dev/null || true

dump_failing_logs() {
  echo "::error::$NAME (pytorch gfx950) FAILED — full per-run logs below"
  local logs f rel
  logs=$(find "$SCRATCH" -name run.log -path '*/run*/*' 2>/dev/null | sort)
  [ -z "$logs" ] && logs=$(find "$SCRATCH" -name run.log 2>/dev/null | sort)
  [ -z "$logs" ] && { echo "::warning::no per-run run.log under $SCRATCH"; return; }
  for f in $logs; do
    rel="${f#$SCRATCH/}"
    echo "::group::run.log — $rel"; cat "$f" 2>/dev/null || echo "(unreadable)"; echo "::endgroup::"
  done
}

SJ=$(find "$SCRATCH" -name summary.json 2>/dev/null | sort | tail -1 || true)
# The pytorch harness exits non-zero on expected equivalence divergence, but
# still writes summary.json. So the gate is evaluated from summary.json, NOT the
# make exit code; only treat a MISSING summary as a hard failure (crashed before
# completion).
if [ -z "$SJ" ]; then
  write_result fail "no summary.json (make exit=$docker_rc; crashed before writing)"
  echo ":x: \`$NAME\` ($LANE): run FAILED — no summary.json (exit=$docker_rc)." >> "$GITHUB_STEP_SUMMARY"
  dump_failing_logs
  exit 1
fi

VERDICT=$(python3 "$GATE" "$SJ"); rc=$?
# VERDICT = "<ok>|proof=..|equiv=<status>|equiv_passed=<bool>"
eq_status=$(echo "$VERDICT" | sed -n 's/.*equiv=\([^|]*\).*/\1/p')
eq_passed=$(echo "$VERDICT" | sed -n 's/.*equiv_passed=\([^|]*\).*/\1/p')
# Tri-state: gate FAIL -> fail; gate PASS with divergence (equiv_passed not True
# or status in {diverged,output_mismatch}) -> diverged (yellow); else pass.
# numerically_close / equivalent / distributionally_equivalent = PASS.
if [ "$rc" != "0" ]; then
  state=fail
elif [ "$eq_status" = "diverged" ] || [ "$eq_status" = "output_mismatch" ] || [ "$eq_passed" != "True" ]; then
  state=diverged
else
  state=pass
fi
detail=$(echo "$VERDICT" | cut -d'|' -f2-)
write_result "$state" "$detail"
case "$state" in pass) EMO=":white_check_mark:";; diverged) EMO=":warning:";; fail) EMO=":x:";; skip) EMO=":fast_forward:";; *) EMO=":page_facing_up:";; esac
echo "${EMO} \`$NAME\` ($LANE): $state — $detail (see consolidated summary)." >> "$GITHUB_STEP_SUMMARY"
[ "$rc" = "0" ] || dump_failing_logs
exit $rc
