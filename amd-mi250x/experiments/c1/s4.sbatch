#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_s4
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=30G
#SBATCH -t 00:30:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/s4_%j.out
srun /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "bash /scratch/project_XXXXXXXXX/tilelang/exp/C1/s4_job.sh"
