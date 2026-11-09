#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/D128/out/slurm-%j.out
# $1 order. plan2 in the fork process, native wheel path in the wheel process. Ratio: ours / same-process vendor, TF, higher is better.
set -uo pipefail
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/D128/src/env.sh
O=$D/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
date +%T; md5sum $SRC/d128b.py $SRC/wport2.py $SRC/tlfa.py $SRC/gqa_fa.py $SRC/plan2.json $F/build/lib/libtilelang.so
nodetag
FM $D/cache/F "d128b.py --tag ${SLURM_JOB_ID}_$1 --order $1 --inputs randn0,big,ar" 2>> err.log | grep '^{' > race.jsonl; echo "rc=${PIPESTATUS[0]}"; date +%T
WH $D/cache/W "wport2.py --tag ${SLURM_JOB_ID}_w --order $1" 2>> err.log | grep '^{' > wport2.jsonl; echo "rc=${PIPESTATUS[0]}"
date +%T; echo DONE
