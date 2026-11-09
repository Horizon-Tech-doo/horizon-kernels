#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_s3
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/s3_%j.out
srun /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "bash /scratch/project_XXXXXXXXX/tilelang/exp/C1/s3_job.sh"
