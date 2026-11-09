#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_s12
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=60G
#SBATCH -t 00:08:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/s12_%j.out
srun /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "bash /scratch/project_XXXXXXXXX/tilelang/exp/C1/s12_job.sh"
