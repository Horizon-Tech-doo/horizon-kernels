#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_s19
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=60G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/raw/s19/s19_%j.out
srun /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "bash /scratch/project_XXXXXXXXX/tilelang/exp/C1/s19_job.sh"
