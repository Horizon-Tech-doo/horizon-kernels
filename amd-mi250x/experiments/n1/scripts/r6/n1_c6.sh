#!/bin/bash
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
C=/tmp/n1c6_$$
timeout 900 singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton PYTHONPATH=$VP CUDA_VISIBLE_DEVICES=; cd $E; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); sys.argv=[\"$D/n1_c6.py\",\"8192\",\"${1:-all}\"]; runpy.run_path(\"$D/n1_c6.py\",run_name=\"__main__\")'" 2>&1 | grep -v "INFO\|Warning\|in memory cache\|amdsmi\|raw_cnt" > raw/c6_T8192_${1:-all}.jsonl
rm -rf $C; echo done >> raw/c6_T8192_${1:-all}.jsonl
