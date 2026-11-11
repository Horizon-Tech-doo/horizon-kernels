#!/bin/bash
#SBATCH --job-name=p1-subnormal
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --time=00:10:00
TL=/scratch/project_XXXXXXXXX/tilelang; E=$TL/exp/P1; SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
export TILELANG_CACHE_DIR=$TL/exp/PROFILE/cache-tl
md5sum $E/vi3/p1_subnormal.py | sed "s# .*/# #"
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$TL/exp/PROFILE/vi/gm-src:$TL/venv/lib/python3.12/site-packages; python3 $E/vi3/p1_subnormal.py" 2>&1 | grep -v "^W1\|agent.cpp\|TileLang" | cut -c1-700
echo "## job end $(date +%T)"
