#!/bin/bash
#SBATCH --job-name=n3-dec
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-final-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; CY=${1:-3}; ORDER=${2:-"lumi image"}
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
lumi(){ srun --ntasks=1 /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python3 n3_bench.py lumi $CY" 2>&1 | grep -v "Warning\|INFO"; }
img(){ srun --ntasks=1 singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 n3_bench.py image $CY" 2>&1 | grep -v "Warning\|INFO\|amdsmi\|raw_cnt"; }
echo "== job $SLURM_JOB_ID order $ORDER $(date +%T)"; rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
for s in $ORDER; do echo "## $s $(date +%T)"; if [ $s = lumi ]; then lumi; else img; fi; done 2>&1 | sed "s/nid[0-9]*/NODE/g" | tee raw/final-$SLURM_JOB_ID.jsonl
echo "## landed script $(date +%T)"; srun --ntasks=1 /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python3 gqa_decode.py paged" 2>&1 | grep -v "INFO\|Warning" | sed "s/nid[0-9]*/NODE/g" > raw/landed-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
