#!/bin/bash
#SBATCH --job-name=gm-tiles
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
# PROFILE / GEMM tile sweep on one GCD for the default prefill chunk (M 1568, 784): gemm_bf16 forms against the server path. GM_REV=1 for the reversed-order job; GM_ARGS="<M list> <N:K list>".
set -uo pipefail
PROJ=${PROJ:-project_XXXXXXXXX}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}; E=$TL/exp/PROFILE; VI=$E/vi; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; SITE=$TL/venv/lib/python3.12/site-packages
export TILELANG_CACHE_DIR=$E/cache-tl-bf16; mkdir -p $TILELANG_CACHE_DIR
{ echo "## job $J rev=${GM_REV:-0} args ${GM_ARGS:-default}"; md5sum $VI/gm-src/gemm_prefetch.py $VI/gm-src/gemm_bf16.py $VI/gm_tiles.py $VI/gm_tiles.sbatch | sed "s# .*/# #"; } | tee $O/header.txt
timeout 55m singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/gm-src:$SITE GM_REV=${GM_REV:-0}; python3 $VI/gm_tiles.py $O/tiles.jsonl ${GM_ARGS:-}" > $O/tiles.log 2>&1; echo "## probe rc=$? $(date +%T)"; grep -c '"kind": "cell"' $O/tiles.jsonl; grep -h "Traceback\|Error\|error" $O/tiles.log | cut -c1-300 | sort | uniq -c | head -8; tail -1 $O/tiles.jsonl | cut -c1-400
echo "## job end $(date +%T)"
