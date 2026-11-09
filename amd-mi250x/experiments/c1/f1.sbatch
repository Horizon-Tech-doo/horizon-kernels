#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C1_fork_build
#SBATCH -p small
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 00:45:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C1/f1_%j.out
/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "bash /scratch/project_XXXXXXXXX/tilelang/exp/C1/f1_build.sh"
