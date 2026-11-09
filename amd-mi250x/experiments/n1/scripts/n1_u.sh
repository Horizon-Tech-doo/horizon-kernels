#!/bin/bash
# unroll trace: k1/k2 by stack and threshold, then dump code objects, then run every code object in both stacks.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; cd $E; mkdir -p co raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
VENV="export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP"
bash n1_job.sh ${1:-3} --set u --rounds 3 --rep 50 --tokens 4096,1024 --arms 0:0,20000:0,0:20000,20000:20000
$TL "cd $E; python n1dump.py $E/cache c18 $E/co" 2>&1 | grep -v "INFO\|Warning"
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "$VENV; cd $E; python3 n1dump.py $E/cache-v c22 $E/co" 2>&1 | grep -v "INFO\|Warning\|amdsmi"
for c in 1 2; do
$TL "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python n1x.py $E/data.pt lumi $E/co" 2>&1 | grep -v "INFO\|Warning" | tee -a raw/x_$SLURM_JOB_ID.jsonl
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "$VENV; cd $E; python3 n1x.py $E/data.pt vllm $E/co" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" | tee -a raw/x_$SLURM_JOB_ID.jsonl
done
date
