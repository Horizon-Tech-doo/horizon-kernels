#!/bin/bash
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; cd $E
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
VENV="export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP"
for c in 1 2; do
$TL "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E/v3; python n1x.py $E/data.pt lumi $E/co_u" 2>&1 | grep -v "INFO\|Warning" | tee -a raw/x_$SLURM_JOB_ID.jsonl
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "$VENV; cd $E/v3; python3 n1x.py $E/data.pt vllm $E/co_u" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" | tee -a raw/x_$SLURM_JOB_ID.jsonl
done
date
