#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/D128/out/slurm-%j.out
# $1 order. Wheel process only: stub wrapper plus frozen bytes against landed native.
set -uo pipefail
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/D128/src/env.sh
O=$D/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
date +%T; md5sum $SRC/wstub.py $SRC/gqa_fa.py
nodetag
WH $D/cache/W "wstub.py --tag ${SLURM_JOB_ID}_$1 --order $1" 2>> err.log | grep '^{' > wstub.jsonl; echo "rc=${PIPESTATUS[0]}"
date +%T; echo DONE
