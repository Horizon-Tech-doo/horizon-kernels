#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --job-name=n5cap
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out3/slurm-%j.out
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
date
singularity exec -B "$TL_BIND" "$IMG" bash -c "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; cd $S; python3 capture_real.py $E/real 4096,8192,16384 3,15,31"
date; echo DONE
