#!/bin/bash
#SBATCH --job-name=gm-skinny
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
# PROFILE / GEMM gate 3 on one GCD: gemm_bf16.py (bf16 MFMA form of the hand kernels) against the server's dense-linear path. GM_REV=1 for the reversed-order job; GM_ARGS="<M list> <N:K list>".
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}; E=$TL/exp/PROFILE; VI=$E/vi; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; SITE=$TL/venv/lib/python3.12/site-packages
export TILELANG_CACHE_DIR=$E/cache-tl-bf16; mkdir -p $TILELANG_CACHE_DIR
{ echo "## job $J rev=${GM_REV:-0} args ${GM_ARGS:-default}"; md5sum $VI/gm-src/gemm_prefetch.py $VI/gm-src/gemm_bf16.py $VI/gm_skinny.py $VI/gm_skinny.sbatch | sed "s# .*/# #"; } | tee $O/header.txt
timeout 55m singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/gm-src:$SITE GM_REV=${GM_REV:-0}; python3 $VI/gm_skinny.py $O/skinny.jsonl ${GM_ARGS:-}" > $O/skinny.log 2>&1; echo "## probe rc=$? $(date +%T)"; grep -c '"kind": "cell"' $O/skinny.jsonl; grep -h "Traceback\|Error\|error" $O/skinny.log | cut -c1-300 | sort | uniq -c | head -8; tail -1 $O/skinny.jsonl | cut -c1-400
echo "## job end $(date +%T)"
