#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_s16
#SBATCH -p small-g
#SBATCH --gpus-per-node=1
#SBATCH -c 7
#SBATCH --mem=60G
#SBATCH -t 00:04:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/s16_%j.out
srun singularity exec -B /pfs,/scratch,/flash /flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif bash /scratch/project_XXXXXXXXX/tilelang/exp/C1/s16_job.sh
