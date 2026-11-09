#!/bin/bash
#SBATCH --job-name=n3-r3w
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=30G
#SBATCH --time=00:04:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; ROT=${1:-0}
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1; cd $E; python3 n3r3w.py $ROT 3" 2>&1 | grep -v "Warning\|INFO\|WARNING\|amdsmi\|raw_cnt" | sed "s/nid[0-9]*/NODE/g" > raw/r3w-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
