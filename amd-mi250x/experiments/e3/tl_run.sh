#!/usr/bin/env bash
# Usage: tl_run.sh "<shell command>"  -- runs it inside the LUMI PyTorch container with the TileLang venv active.
source "$(dirname "$(readlink -f "$0")")/env.sh"
exec singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$TL_ROOT/cache TRITON_CACHE_DIR=$TL_ROOT/cache/triton; $1"
