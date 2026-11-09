#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -J C2P_b2
#SBATCH -p small
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 00:20:00
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/b2_%j.out
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
singularity exec -B "$TL_BIND" "$IMG" bash /scratch/project_XXXXXXXXX/tilelang/exp/C2P/r5/b2_build.sh
