#!/bin/bash
#SBATCH --job-name=n3-skept
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; CY=${1:-5}
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
echo "== job $SLURM_JOB_ID $(date +%T)"
srun --ntasks=1 singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 n3_skeptic.py $CY" 2>&1 | grep -v "Warning\|INFO\|amdsmi\|raw_cnt" | sed "s/nid[0-9]*/NODE/g" > raw/skeptic-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
