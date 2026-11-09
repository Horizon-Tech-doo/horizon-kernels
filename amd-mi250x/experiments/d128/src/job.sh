#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/D128/out/slurm-%j.out
# $1 fwd|rev, $2 regex of arms, $3 inputs. Ratio: ours / same-process vendor, TF, higher is better.
set -uo pipefail
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/D128/src/env.sh
O=$D/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
date +%T; md5sum $SRC/d128.py $SRC/tlfa.py $SRC/plan.json $F/build/lib/libtilelang.so | cut -c1-32,33-
nodetag
FM $D/cache/F "d128.py --tag ${SLURM_JOB_ID}_$1 --order $1 --re '$2' --inputs $3" 2>> err.log | grep '^{' > race.jsonl; echo "rc=${PIPESTATUS[0]}"
date +%T; echo DONE
