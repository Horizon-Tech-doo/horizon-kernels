#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C2P_b0
#SBATCH -p small
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/b0_%j.out
/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "bash /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/b0_build.sh"
