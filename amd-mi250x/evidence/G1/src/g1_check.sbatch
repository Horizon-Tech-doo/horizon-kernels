#!/bin/bash
#SBATCH --job-name=g1-check
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:40:00
# G1 gate, one GCD, in the server's own environment: compile horizon_lumi_gemm's kernels, then g1_check.py (error against fp32, refusals, padding, non-finite inputs, timings against vLLM's path).
set -uo pipefail
PROJ=${PROJ:-project_XXXXXXXXX}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}; E=$TL/exp/G1; VI=${VI:-$E/vi}; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; SITE=$TL/venv/lib/python3.12/site-packages
export HLG_SCRIPTS=$VI/gm-src TILELANG_CACHE_DIR=$E/cache-tl HLG_MS=${HLG_MS:-1568} HLG_SHAPES=${HLG_SHAPES:-17408x5120,5120x8704,8192x5120,7168x5120,5120x3072} HLG_FORMS="${HLG_FORMS:-}"; mkdir -p $TILELANG_CACHE_DIR
{ echo "## job $J HLG_MS=$HLG_MS HLG_SHAPES=$HLG_SHAPES HLG_FORMS=$HLG_FORMS VI=$VI"; md5sum $VI/gm-src/gemm_prefetch.py $VI/gm-src/gemm_bf16.py $VI/overlay/horizon_lumi_gemm/__init__.py $VI/g1_check.py $VI/g1_precompile.py $VI/g1_check.sbatch | sed "s# .*/# #"; } | tee $O/header.txt
X="singularity exec -B /pfs,/scratch,/flash $SIF bash -c"
echo "## compile start $(date +%T)"
for i in 0 1 2 3 4 5 6; do $X "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/g1_precompile.py $i 7" > $O/compile-$i.log 2>&1 & done; wait
echo "## compile end $(date +%T) compiled $(grep -h '^compiled' $O/compile-*.log | wc -l)"; grep -h 'Traceback\|Error' $O/compile-*.log | cut -c1-240 | sort | uniq -c | tail -8
timeout 25m $X "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/g1_check.py $O/check.jsonl" > $O/check.log 2>&1; echo "## check rc=$? $(date +%T)"; grep -c '"kind": "cell"' $O/check.jsonl; tail -1 $O/check.jsonl | cut -c1-700; grep -h "Traceback\|Error" $O/check.log | cut -c1-240 | sort | uniq -c | tail -5
echo "## job end $(date +%T)"
