#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/D128/out/slurm-%j.out
# $1 first order, $2 second order: 8k and 16k, six inputs, one process per order, same GCD, so order and instance separate; then the wheel-process check in the first order. Ratio: ours / same-process vendor, TF, higher is better.
set -uo pipefail
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/D128/src/env.sh
O=$D/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
date +%T; md5sum $SRC/d128.py $SRC/wport.py $SRC/tlfa.py $SRC/gqa_fa.py $SRC/plan.json $F/build/lib/libtilelang.so
nodetag
for o in $1 $2; do FM $D/cache/F "d128.py --tag ${SLURM_JOB_ID}_$o --order $o --re '_(8192|16384)_' --inputs randn0,randn1,self,big,ar,ramp" 2>> err.log | grep '^{' > race_$o.jsonl; echo "rc=${PIPESTATUS[0]}"; date +%T; done
WH $D/cache/W "wport.py --tag ${SLURM_JOB_ID}_w --order $1" 2>> err.log | grep '^{' > wport.jsonl; echo "rc=${PIPESTATUS[0]}"
date +%T; echo DONE
