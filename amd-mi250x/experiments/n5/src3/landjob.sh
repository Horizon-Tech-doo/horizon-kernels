#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=n5land3
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out3/slurm-%j.out
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out3/$SLURM_JOB_ID; mkdir -p $O; cd $O
N5_ENV=""
date; md5sum $S/gqa_fa.py /scratch/project_XXXXXXXXX/tilelang/scripts/gqa_fa.py
TL_I land3 "land3.py" 2>> err.log | grep '^{' | tee -a run.jsonl
TL_L land3 "land3.py" 2>> err.log | grep '^{' | tee -a run.jsonl
TL_I land3 "gqa_fa.py 4096 - 128" 2>> err.log | grep '^{' | tee main_d128.jsonl
date; echo DONE
