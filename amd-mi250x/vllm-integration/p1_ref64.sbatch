#!/bin/bash
#SBATCH --job-name=p1-ref64
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:50:00
# P1 follow-up on one GCD: prefill attention against a chunked float64 reference up to 16,384 tokens, and the cost of the plugin-only cached-context form (square kernel + cache gather).
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}; E=$TL/exp/P1; VI=$E/vi; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; SITE=$TL/venv/lib/python3.12/site-packages
export HLP_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl HLP_LADDER=1024:16384:0.06
md5sum $HLP_SCRIPTS/gqa_fa.py $VI/overlay/horizon_lumi_prefill/__init__.py $VI/p1_ref64.py $VI/p1_ref64.sbatch | sed "s# .*/# #" | tee $O/header.txt
timeout 45m singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/p1_ref64.py $O/ref64.jsonl" > $O/ref64.log 2>&1; echo "## rc=$? $(date +%T)"; grep -c '"kind"' $O/ref64.jsonl; grep -h "Traceback\|Error" $O/ref64.log | cut -c1-300 | sort | uniq -c | tail -5; tail -1 $O/ref64.jsonl | cut -c1-300
echo "## job end $(date +%T)"
