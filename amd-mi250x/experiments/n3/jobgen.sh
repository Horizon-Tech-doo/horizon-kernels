#!/bin/bash
#SBATCH --job-name=n3-dec
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=30G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-gen-%j.out
srun --ntasks=1 /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd /scratch/project_XXXXXXXXX/tilelang/exp/N3; python3 gen_bases.py" 2>&1 | grep -v "Warning\|INFO"
