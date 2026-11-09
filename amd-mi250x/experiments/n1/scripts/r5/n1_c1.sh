#!/bin/bash
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v9; cd $E; mkdir -p raw asm_r5
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
T=${1:-8192}
C=/tmp/n1pol_v_$$; CL=/tmp/n1pol_l_$$
timeout 600 singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton PYTHONPATH=$VP CUDA_VISIBLE_DEVICES=; cd $E; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); sys.argv=[\"$D/n1pol.py\",\"$E/asm_r5/c22\",\"$T\",\"image\"]; runpy.run_path(\"$D/n1pol.py\",run_name=\"__main__\")'" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" | tee raw/pol_c22_T$T.jsonl
timeout 600 $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton CUDA_VISIBLE_DEVICES=; cd $E; python -I -c 'import sys,runpy; sys.path.insert(0,\"$D\"); sys.argv=[\"$D/n1pol.py\",\"$E/asm_r5/c18\",\"$T\",\"lumi\"]; runpy.run_path(\"$D/n1pol.py\",run_name=\"__main__\")'" 2>&1 | grep -v "INFO\|Warning\|in memory cache" | tee raw/pol_c18_T$T.jsonl
rm -rf $C $CL
