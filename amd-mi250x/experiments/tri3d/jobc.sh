#!/bin/bash
#SBATCH --job-name=tri3d-c
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/TRI3D/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/TRI3D; cd $E; SCR=tri3d.py
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
image(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 $SCR image $*" 2>&1 | grep -v "Warning\|INFO\|WARNING\|amdsmi\|raw_cnt"; }
{ md5sum /scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py $E/$SCR; echo "## job $SLURM_JOB_ID compile start $(date +%T)"
  for i in 0 1 2 3 4 5; do image 0 1 compile $i 6 & done; wait; echo "## compiled $(date +%T)"
} 2>&1 | sed "s/nid[0-9]*/<node>/g" | cut -c1-600 > raw/c-$SLURM_JOB_ID.log
