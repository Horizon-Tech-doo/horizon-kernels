#!/bin/bash
# Usage: sbatch --job-name=E3-x ... job.sh <case> <regex> <rounds> ; interleaves TL and CK passes in one allocation.
set -uo pipefail
E=/scratch/project_XXXXXXXXX/tilelang/exp/E3
S=$E/src6; O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
case=$1; re=$2; rounds=$3
IMG=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
source $S/env.sh
TL() { timeout "$1" singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache6$3 TRITON_CACHE_DIR=$E/cache/triton E3_PATCH=$3; cd $S; $2"; }
CK() { SINGULARITYENV_PYTHONNOUSERSITE=1 timeout "$1" singularity exec -B /pfs,/scratch,/flash $IMG bash -c "cd $S; python3 $2"; }
( while true; do date +%T; rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -2; rocm-smi --showpower 2>/dev/null | grep -iE "power" | head -1; sleep 20; done ) > clocks.log 2>&1 &
CLK=$!
trap "kill $CLK 2>/dev/null" EXIT
hostname; date
for r in $(seq 1 $rounds); do
  chk=""; [ "$r" = 1 ] && chk="--check"
  for fl in "" KXABD; do
    echo "=== round $r TL[$fl] $(date +%T)"
    TL 500 "python run_tle.py $case '$re' r$r $chk" "$fl" >> tl_${r}_$fl.jsonl 2>> tl_${r}_$fl.err; echo "tl rc=$?"
  done
  echo "=== round $r CK $(date +%T)"
  CK 500 "run_ck.py $case r$r $chk" >> ck_$r.jsonl 2> ck_$r.err; echo "ck rc=$?"
done
date; echo DONE
