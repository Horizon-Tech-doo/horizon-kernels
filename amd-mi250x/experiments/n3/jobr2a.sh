#!/bin/bash
#SBATCH --job-name=n3-r2a
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --time=00:07:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; ORDER=${1:-"image lumi"}; SCR=${2:-n3r2a.py}; TAG=${3:-r2a}; shift 3
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
lumi(){ srun --ntasks=1 /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python3 $SCR lumi $*" 2>&1 | grep -v "Warning\|INFO"; }
img(){ srun --ntasks=1 singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 $SCR image $*" 2>&1 | grep -v "Warning\|INFO\|amdsmi\|raw_cnt"; }
echo "== job $SLURM_JOB_ID order $ORDER $(date +%T)"
for s in $ORDER; do echo "## $s $(date +%T)"; if [ $s = lumi ]; then lumi "$@"; else img "$@"; fi; done 2>&1 | sed "s/nid[0-9]*/NODE/g" > raw/$TAG-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
