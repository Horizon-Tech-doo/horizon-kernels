#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_s5
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=60G
#SBATCH -t 00:25:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/s5_%j.out
srun /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd /scratch/project_XXXXXXXXX/tilelang/exp/C1; mkdir -p raw; PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/fork LD_LIBRARY_PATH=/scratch/project_XXXXXXXXX/tilelang/fork/build/lib:\$LD_LIBRARY_PATH TILELANG_CACHE_DIR=\$PWD/cache_fork python s5_sweep.py 3 2>&1 | grep -E '^\{|Error' > raw/s5_\$SLURM_JOB_ID.jsonl"
