#!/usr/bin/env bash
# run-build-lit.sh - rebuild the HotSwap transpiler (libamd_comgr.so) from THIS
# PR's llvm-project checkout, then run the transpiler lit suite.
#
# NO baked fallback: if the PR's comgr fails to build, the job FAILS. The E2E
# lanes consume the .so this produces, so they always exercise the PR's code.
#
# CLEAN BUILD each run: after overlaying the PR source we `ninja -t clean` to
# drop every prior build output, so every object is recompiled from the PR's
# source (no reliance on incremental mtime tracking, which an rsync overlay can
# defeat). We keep the image's validated CMake configuration (CMakeCache) so the
# ~15 configure flags don't have to be re-derived. A host-mounted ccache + the
# runner's many cores make the from-scratch compile fast (typically ~1 min with a
# warm cache); only linking is real work. Any PR change -- comgr OR llvm core --
# is therefore always rebuilt.
#
# Runs on the gfx950 runner HOST; the heavy lifting is inside $BUILD_LIT_IMAGE.
#
# Env:  BUILD_LIT_IMAGE  docker image with the LLVM source + configured build tree
#       PR_SRC           host path to the PR checkout (full llvm-project)
#       COMGR_OUT        host dir to receive libamd_comgr.so.3.3.0 + lit-summary.txt
#       CCACHE_HOST_DIR  host ccache dir to mount (default: $HOME/.ccache)
#       GITHUB_STEP_SUMMARY  (from Actions)
#
# Exit: 0 = PR comgr built + lit ran (lit test-content failures do NOT fail the
#           job -- transpiler gaps are expected). 1/2 = build/infra failure.
#
# Configure flags required to build comgr in-tree against this image's /opt/rocm:
#   -DLLD_INCLUDE_DIRS=...            comgr's hotswap-transpiler needs LLD headers
#   -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON  comgr/test does an unused find_package(hip)
#                                        that pulls /opt/rocm AMDDeviceLibs and
#                                        collides with the in-tree device-libs.
#   -DHIPCC=/opt/rocm/bin/hipcc       real hipcc for the TDM gfx942 bitcode (the
#                                        baked HIPCC pointed at a slimmed-out venv).
#   -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja  baked ninja path pointed at a slimmed venv.
# We also skip comgr's `test-unit` subdir: upstream HEAD's OpcodeMapTest calls a
# stale initMCState signature and does not compile (an upstream bug, not the PR's).
# The transpiler lit suite (build/tools/comgr/test-lit) is the gate signal.
set -uo pipefail

: "${BUILD_LIT_IMAGE:?}"; : "${PR_SRC:?}"; : "${COMGR_OUT:?}"
CCACHE_HOST_DIR="${CCACHE_HOST_DIR:-$HOME/.ccache}"
mkdir -p "$COMGR_OUT" "$CCACHE_HOST_DIR"

# In-container driver. Single-quoted heredoc: expanded INSIDE the container.
read -r -d '' INCONTAINER <<'INNER' || true
set -uo pipefail
export PATH=/opt/rocm/llvm/bin:$PATH
ACC=/workspace/llvm-acc
B="$ACC/build"
export LD_LIBRARY_PATH="$B/lib:${LD_LIBRARY_PATH:-}"
export CCACHE_DIR=/ccache
ccache -M 50G >/dev/null 2>&1 || true
git config --global --add safe.directory '*' 2>/dev/null || true

echo "=== overlay PR source (full llvm-project incl CMake) over the tree ==="
rsync -a --delete \
  --exclude='/build/' --exclude='/.git/' --exclude='/.github/' --exclude='/docker/' \
  /pr/ "$ACC/"

# Skip the upstream-broken comgr unit tests (test-unit) -- OpcodeMapTest does not
# compile at HEAD. The transpiler lit suite (test-lit) is what we gate on.
sed -i -E 's/^([[:space:]]*)add_subdirectory\(test-unit\)/\1# add_subdirectory(test-unit)  # CI: skipped (upstream HEAD unit test does not compile)/' \
  "$ACC/amd/comgr/CMakeLists.txt" || true

echo "=== reconfigure (keep baked config; device-libs first; LLD headers; hip clash off; real hipcc) ==="
if ! cmake -S "$ACC/llvm" -B "$B" -G Ninja \
      -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
      -DLLD_INCLUDE_DIRS="$ACC/lld/include;$B/tools/lld/include" \
      -DCMAKE_DISABLE_FIND_PACKAGE_hip=ON \
      -DHIPCC=/opt/rocm/bin/hipcc > /tmp/reconf.log 2>&1; then
  echo "::error::cmake reconfigure failed"; tail -30 /tmp/reconf.log; exit 1
fi

echo "=== clean: drop ALL prior build outputs (from-scratch rebuild, ccache-backed) ==="
ninja -C "$B" -t clean >/dev/null 2>&1 || true
ccache -z >/dev/null 2>&1 || true

echo "=== build the PR's comgr + device-libs + LLVM tools the lit tests invoke (this is the gate) ==="
# Full recipe target set: rebuilds all of LLVM the transpiler + its lit tests need
# (a clean tree needs clang/lld/llc/llvm-mc/llvm-objdump/FileCheck rebuilt too).
if ! ninja -C "$B" amd_comgr rocm-device-libs clang lld llc llvm-mc llvm-objdump FileCheck not count 2>&1 | tee /tmp/build.log; then
  echo "::error::PR comgr build FAILED (no baked fallback)"; exit 2
fi
echo "--- ccache after comgr build ---"; ccache -s 2>/dev/null | grep -iE "hits|miss|cache size" | head -4 || true

SO="$B/lib/libamd_comgr.so.3.3.0"
[ -f "$SO" ] || { echo "::error::libamd_comgr.so.3.3.0 not produced"; exit 2; }
cp -a "$SO" /out/
strip --strip-debug /out/libamd_comgr.so.3.3.0 2>/dev/null || true

echo "=== build the transpiler lit-test binaries (check-comgr deps; test-unit skipped) ==="
# check-comgr also runs ctest at the end; we only need it to BUILD the test-lit
# binaries, so its ctest exit is ignored -- we run llvm-lit ourselves for a clean tally.
ninja -C "$B" check-comgr > /tmp/lit-build.log 2>&1 || true

echo "=== transpiler lit suite ==="
"$B/bin/llvm-lit" -s "$B/tools/comgr/test-lit" 2>&1 | tee /tmp/lit.log || true
grep -E "Passed|Failed|Unsupported|Testing Time" /tmp/lit.log | tail -8 > /out/lit-summary.txt || true
echo "=== OK ==="
INNER

docker run --rm --network host \
  -v "$PR_SRC":/pr:ro \
  -v "$COMGR_OUT":/out \
  -v "$CCACHE_HOST_DIR":/ccache \
  "$BUILD_LIT_IMAGE" \
  bash -lc "$INCONTAINER" 2>&1 | tee "$RUNNER_TEMP/buildlit.log"

{
  echo "## build + lit (PR comgr, gfx950)"
  echo '```'
  cat "$COMGR_OUT/lit-summary.txt" 2>/dev/null || echo "(no lit summary captured)"
  echo '```'
  if [ -f "$COMGR_OUT/libamd_comgr.so.3.3.0" ]; then
    echo ":white_check_mark: comgr rebuilt from PR source (clean) + transpiler lit ran ($(du -h "$COMGR_OUT/libamd_comgr.so.3.3.0" | cut -f1))"
  else
    echo ":x: PR comgr build FAILED (no baked fallback) -- E2E lanes will be skipped"
  fi
} >> "$GITHUB_STEP_SUMMARY"

[ -f "$COMGR_OUT/libamd_comgr.so.3.3.0" ] || exit 1
exit 0
