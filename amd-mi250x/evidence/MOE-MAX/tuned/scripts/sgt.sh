#!/bin/bash
# sgt.sh SRCDIR CACHE script args... : run a script from SRCDIR in the vLLM image (login or job). Same as round 6's sg6.sh.
E=/scratch/project_XXXXXXXXX/tilelang/exp/MOE-MAX; D=$1; shift; C=$1; shift; S=$1; shift
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
mkdir -p $C; cd $E
singularity exec -B /pfs,/scratch,/flash $SIF env VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP $SGT_ENV \
  python3 -I -c 'import sys,runpy; sys.path.insert(0,sys.argv[1]); sys.path.insert(0,sys.argv[2]); sys.argv=sys.argv[3:]; runpy.run_path(sys.argv[0], run_name="__main__")' $VP $D $D/$S "$@" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache\|agent.cpp"
