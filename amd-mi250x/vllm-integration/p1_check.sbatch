#!/bin/bash
#SBATCH --job-name=p1-check
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --output=%x-%j.out
# P1 gate, one GCD, in the server's own environment (the vLLM image + the TileLang venv): compile every HLP bucket at Hkv 2, then p1_check.py (error against float64, guard cases, timings with padding and casts).
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}; E=$TL/exp/P1; VI=$E/vi; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; SITE=$TL/venv/lib/python3.12/site-packages
export HLP_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl HLP_LADDER=${HLP_LADDER:-1024:16384:0.06}
md5sum $HLP_SCRIPTS/gqa_fa.py $VI/overlay/horizon_lumi_prefill/__init__.py $VI/p1_check.py $VI/p1_precompile.py | sed "s# .*/# #" | tee $O/header.txt
X="singularity exec -B /pfs,/scratch,/flash $SIF bash -c"
echo "## compile start $(date +%T)"
for i in 0 1 2 3 4 5 6; do $X "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/p1_precompile.py $i 7 2" > $O/compile-$i.log 2>&1 & done; wait
echo "## compile end $(date +%T) compiled $(grep -h '^compiled' $O/compile-*.log | wc -l)"; grep -h 'Traceback\|Error' $O/compile-*.log | cut -c1-240 | sort | uniq -c | tail -8; grep -h '^compiled' $O/compile-*.log | sort -k6 -n | tail -2
timeout 20m $X "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/p1_check.py $O/check-hkv2.jsonl 2" > $O/check-hkv2.log 2>&1; echo "## check hkv2 rc=$? $(date +%T)"; tail -3 $O/check-hkv2.log | cut -c1-600
[ -n "${HKV4:-}" ] && { timeout 12m $X "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/p1_check.py $O/check-hkv4.jsonl 4 $HKV4" > $O/check-hkv4.log 2>&1; echo "## check hkv4 rc=$? $(date +%T)"; tail -2 $O/check-hkv4.log | cut -c1-600; }
echo "## job end $(date +%T)"
