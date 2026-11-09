#!/bin/bash
#SBATCH --job-name=n3-r2c
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; ORDER=${1:-"image lumi"}; TAIL=${2:-kvg}; CY=${3:-3}; PRE=${4:-0}; WT=${5:-6}; SCR=${6:-n3r2c.py}
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
lumi(){ /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python3 $SCR lumi $*" 2>&1 | grep -v "Warning\|INFO\|WARNING"; }
img(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 $SCR image $*" 2>&1 | grep -v "Warning\|INFO\|WARNING\|amdsmi\|raw_cnt"; }
echo "== job $SLURM_JOB_ID order $ORDER $(date +%T)"
for s in $ORDER; do echo "## $s $(date +%T)"; f=$( [ $s = lumi ] && echo lumi || echo img )
  if [ "$PRE" = 1 ]; then for i in $(seq 0 $((WT-1))); do $f $TAIL $CY compile $i $WT & done; wait; echo "## $s compiled $(date +%T)"; fi
  $f $TAIL $CY
done 2>&1 | sed "s/nid[0-9]*/NODE/g" > raw/r2c-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
