#!/bin/bash
#SBATCH --job-name=n3-r3z
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=00:08:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
# final bytes: the script entry point on both stacks
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; S=/scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
{ md5sum $S
echo "## lumi"; /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python3 $S paged" 2>&1 | grep -v "INFO\|Warning\|WARNING"
echo "## image"; singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 $S paged" 2>&1 | grep -v "Warning\|INFO\|WARNING\|amdsmi\|raw_cnt"
} 2>&1 | sed "s/nid[0-9]*/NODE/g" > raw/r3z-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
