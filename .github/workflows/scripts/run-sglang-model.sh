#!/usr/bin/env bash
# Run one SGLang HotSwap compare for a single model inside the prebuilt image,
# then evaluate the equivalence gate. Used by the SGLang lanes of hotswap-pr-ci.yml. Runs on the self-hosted runner HOST (not in a container).
#
# Args:   $1 NAME      leg name (e.g. qwen2_5_7b) — used for job + scratch dir
#         $2 PROFILE   harness sglang profile (named, or `custom`)
#         $3 SUBPATH   model dir under $WEIGHTS_HOST (e.g. google/gemma-3-4b-it)
#         $4 PROMPTS   (optional) prompts file for the `custom` profile (in-image path)
# Env:    IMAGE         docker image with the harness + hotswap stack
#         WEIGHTS_HOST  host dir mounted at /data (model is /data/$SUBPATH)
#         SCRATCH_BASE  host dir for per-model scratch
#         GPU           GPU index (e.g. 7)
#         TARGET_GFX    gfx942 | gfx950
#         GITHUB_STEP_SUMMARY, GITHUB_WORKSPACE  (from Actions)
#
# Exit: 0 = gate PASS or weights-absent (skipped); 1 = gate FAIL or run error.
# NOTE: invoke with a shell WITHOUT errexit (find over the root-owned scratch
# dir exits non-zero on permission-denied subpaths).
set -uo pipefail

NAME="$1"; PROFILE="$2"; SUBPATH="$3"; PROMPTS="${4:-}"
HOSTPATH="$WEIGHTS_HOST/$SUBPATH"
SCRATCH="$SCRATCH_BASE/$NAME"
GATE="$GITHUB_WORKSPACE/.github/workflows/scripts/gate-equivalence.py"
# Lane label used by the consolidated summary aggregator. Passed in by the
# workflow via LANE; fall back to a sensible default from TARGET_GFX.
LANE="${LANE:-SGLang E2E ${TARGET_GFX}}"
RESULT_JSON="${RESULT_JSON:-$RUNNER_TEMP/model-result.json}"
# `custom` profile carries no built-in prompts; pass the supplied file.
PROMPTS_ARG=""
[ -n "$PROMPTS" ] && PROMPTS_ARG="SGLANG_PROMPTS=$PROMPTS"

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
  -v "$WEIGHTS_HOST":/data \
  -v "$SCRATCH":/scratch \
  "${PR_COMGR_MNT[@]}" \
  "$IMAGE" \
  bash -lc '
    set -euo pipefail
    cd /workspace/rocm-hotswap-testing
    export LD_LIBRARY_PATH=/workspace/llvm-acc/build/lib:$LD_LIBRARY_PATH
    make sglang-hotswap-e2e-compare \
      SGLANG_MODEL_PATH=/data/'"$SUBPATH"' \
      SGLANG_PYTHON=/workspace/venv/bin/python \
      SGLANG_PROFILE='"$PROFILE"' \
      '"$PROMPTS_ARG"' \
      SGLANG_LIBHSA_RUNTIME=/workspace/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so \
      SGLANG_INTERCEPT_LIB=/workspace/rocm-hotswap-testing/build/libhotswap_intercept.so \
      SGLANG_SCRATCH_ROOT=/scratch \
      SGLANG_GPU='"$GPU"' \
      SGLANG_TARGET_GFX='"$TARGET_GFX"' \
      SGLANG_SOURCE_TARGET=gfx1250:32
  ' 2>&1 | tee "$SCRATCH/run.log"
docker_rc=${PIPESTATUS[0]}
# Hand scratch back to the runner user (container wrote it as root) so this
# script AND the runner's _work/_temp cleanup can delete it next time.
docker run --rm -v "$SCRATCH_BASE":/sb "$IMAGE" chown -R "$(id -u):$(id -g)" "/sb/$NAME" 2>/dev/null || true

# Dump every per-run model log (local/hotswap x run1/run2) to the workflow
# console, each in its own collapsible group, so a failing run's FULL detail is
# visible on the Actions page (not just "run1 failed with exit code 1").
dump_failing_logs() {
  echo "::error::$PROFILE ($TARGET_GFX) FAILED — full per-run logs below"
  # The harness writes detailed logs at <run_dir>/{local,hotswap}/{run1,run2}/run.log
  local logs
  logs=$(find "$SCRATCH" -name run.log -path '*/run*/*' 2>/dev/null | sort)
  [ -z "$logs" ] && logs=$(find "$SCRATCH" -name run.log 2>/dev/null | sort)
  if [ -z "$logs" ]; then
    echo "::warning::no per-run run.log files found under $SCRATCH"
    return
  fi
  local f rel
  for f in $logs; do
    rel="${f#$SCRATCH/}"
    echo "::group::run.log — $rel"
    cat "$f" 2>/dev/null || echo "(could not read $f)"
    echo "::endgroup::"
  done
}

SJ=$(find "$SCRATCH" -name summary.json 2>/dev/null | sort | tail -1 || true)

# Failure if the harness exited non-zero or produced no summary.
if [ "$docker_rc" != "0" ] || [ -z "$SJ" ]; then
  write_result fail "run FAILED (exit=$docker_rc, summary=$( [ -n "$SJ" ] && echo present || echo missing ))"
  echo ":x: \`$NAME\` ($LANE): run FAILED (exit=$docker_rc) — see logs." >> "$GITHUB_STEP_SUMMARY"
  dump_failing_logs
  exit 1
fi

VERDICT=$(python3 "$GATE" "$SJ"); rc=$?
IFS='|' read -r ok status strict <<< "$VERDICT"
# Tri-state: gate FAIL -> fail; gate PASS with a divergence status (or strict
# equivalence not passing) -> diverged (yellow); otherwise -> pass (green).
# numerically_close / equivalent / distributionally_equivalent = PASS.
if [ "$rc" != "0" ]; then
  state=fail
elif [ "$status" = "diverged" ] || [ "$status" = "output_mismatch" ] || [ "$strict" != "True" ]; then
  state=diverged
else
  state=pass
fi
write_result "$state" "overall_status=$status (strict=$strict)"
case "$state" in pass) EMO=":white_check_mark:";; diverged) EMO=":warning:";; fail) EMO=":x:";; skip) EMO=":fast_forward:";; *) EMO=":page_facing_up:";; esac
echo "${EMO} \`$NAME\` ($LANE): $state — overall_status=\`$status\`, strict=$strict (see consolidated summary)." >> "$GITHUB_STEP_SUMMARY"

# On gate failure, also surface the full per-run logs.
[ "$rc" = "0" ] || dump_failing_logs
exit $rc
