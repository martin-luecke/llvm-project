#!/usr/bin/env bash
# build.sh - build the HotSwap build+lit CI image from its Dockerfile.
#
# The CI uses two images from registry-sc-harbor.amd.com/hotswap-ci/:
#   - hotswap-pr10-gfx950-lean      : the prebuilt base, used DIRECTLY as the E2E
#                                     model runner (harness + SGLang/pytorch venv
#                                     + the llvm-acc build tree the transpiler
#                                     needs at runtime). Built imperatively; no
#                                     Dockerfile provenance yet (see Caveats in
#                                     README.md).
#   - hotswap-pr10-gfx950-build-lit : derived HERE from -lean via
#                                     Dockerfile.build-lit; the PR build + lit gate.
#
# Usage:
#   ./build.sh                 # build the build-lit image
#   FLATTEN=0 ./build.sh       # skip export|import flatten (keeps layers; larger
#                              #   reported size but shares the base in Harbor)
#
# Env overrides: BASE (default -lean), REGISTRY (default .../hotswap-ci), FLATTEN (1).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="${BASE:-registry-sc-harbor.amd.com/hotswap-ci/hotswap-pr10-gfx950-lean:latest}"
REGISTRY="${REGISTRY:-registry-sc-harbor.amd.com/hotswap-ci}"
FLATTEN="${FLATTEN:-1}"
TAG="${REGISTRY}/hotswap-pr10-gfx950-build-lit:latest"
TMP="hotswap-build-lit-layered:tmp"

echo ">>> building ${TAG} from Dockerfile.build-lit (BASE=${BASE})"
DOCKER_BUILDKIT=1 docker build \
  --build-arg "BASE=${BASE}" \
  -f "${HERE}/Dockerfile.build-lit" \
  -t "${TMP}" "${HERE}"

if [ "${FLATTEN}" = "1" ]; then
  echo ">>> flattening ${TAG}"
  c="flatten-build-lit-$$"
  docker rm -f "${c}" >/dev/null 2>&1 || true
  docker run -d --name "${c}" "${TMP}" sleep infinity >/dev/null
  # Preserve ENV/WORKDIR/CMD declared in the Dockerfile via inspect.
  env_args=(); e=""
  while IFS= read -r e; do env_args+=(--change "ENV ${e}"); done \
    < <(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${TMP}")
  wd="$(docker inspect --format '{{.Config.WorkingDir}}' "${TMP}")"
  docker export "${c}" | docker import \
    "${env_args[@]}" \
    --change "WORKDIR ${wd}" \
    --change 'CMD ["sleep","infinity"]' \
    - "${TAG}"
  docker rm -f "${c}" >/dev/null
  docker rmi "${TMP}" >/dev/null 2>&1 || true
else
  docker tag "${TMP}" "${TAG}"
  docker rmi "${TMP}" >/dev/null 2>&1 || true
fi
echo ">>> built ${TAG}"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -F "${TAG%%:*}" || true
