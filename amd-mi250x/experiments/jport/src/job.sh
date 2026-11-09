#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/JPORT/out/slurm-%j.out
# $1 fwd|rev. Ratio: ours / same-process vendor, TF, higher is better.
set -uo pipefail
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/JPORT/src/env.sh
O=$X/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
date +%T; md5sum $SRC/new/gqa_fa.py $SRC/jport.py
nodetag
WH $X/cache/W "$SRC/new" "jport.py --tag ${SLURM_JOB_ID}_$1 --order $1" 2>> err.log | grep '^{' > race.jsonl; echo "rc=${PIPESTATUS[0]}"
date +%T; echo DONE
