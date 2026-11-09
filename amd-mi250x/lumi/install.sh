#!/usr/bin/env bash
# Lays this release out the way the scripts expect it on LUMI: one flat scripts directory plus scaling/.
#   export PROJ=project_<your number>     # your LUMI project
#   lumi/install.sh                       # copies into $TL_ROOT (default /scratch/$PROJ/tilelang)
# Then run lumi/setup.sh once (venv, TileLang wheel, hipcc shim). Nothing here starts a job.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
: "${PROJ:?set PROJ to your LUMI project, e.g. export PROJ=project_<number>}"
TL_ROOT=${TL_ROOT:-/scratch/$PROJ/tilelang}
mkdir -p "$TL_ROOT/scripts" "$TL_ROOT/scaling" "$TL_ROOT/logs" "$TL_ROOT/exp/V1/vi/hld-src"
cp "$R"/kernels/*.py "$R"/baselines/*.py "$R"/diagnostics/*.py "$TL_ROOT/scripts/"
cp "$R"/lumi/env.sh "$R"/lumi/setup.sh "$R"/lumi/tl_run.sh "$R"/lumi/gpu_test.sbatch "$R"/lumi/test_reduce.py \
   "$R"/lumi/conftest_rocm_only.py "$TL_ROOT/scripts/"
cp "$R"/scaling/* "$TL_ROOT/scaling/"
# vllm-integration/v1.sbatch reads its overlay, client and the decode kernel from $TL_ROOT/exp/V1/vi.
cp -R "$R"/vllm-integration/. "$TL_ROOT/exp/V1/vi/"
cp "$R"/kernels/gqa_decode.py "$TL_ROOT/exp/V1/vi/hld-src/"
chmod +x "$TL_ROOT"/scripts/*.sh
echo "installed into $TL_ROOT; next: $TL_ROOT/scripts/setup.sh"
