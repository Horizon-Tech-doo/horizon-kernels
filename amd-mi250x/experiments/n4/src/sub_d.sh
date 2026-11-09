#!/bin/bash
# N4 lever D. $1 = a or b (arm order). Prefill and MoE T=16 on both stacks. Copies of landed sources and own caches; nothing under N5 or N1 is written.
set -uo pipefail
L=/scratch/project_XXXXXXXXX/tilelang/exp/N4/levd; cd $L
J=$SLURM_JOB_ID; O=/scratch/project_XXXXXXXXX/tilelang/exp/N4/raw/j_levd_$J; mkdir -p $O; date -u > $O/time.txt; md5sum gqa_fa.py moe_fused.py pre_shift.py moe_shift.py n5lib.py > $O/md5.txt; echo "order=$1" >> $O/time.txt
source /scratch/project_XXXXXXXXX/tilelang/scripts/env.sh
IMG=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=$TL_ROOT/venv/lib/python3.12/site-packages
DATA=/scratch/project_XXXXXXXXX/tilelang/exp/N1/data.pt
tli(){ local t=$1; shift; singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$L/cacheI TRITON_CACHE_DIR=$L/cacheI/triton HF_HUB_OFFLINE=1 N5_STACK=image; cd $L; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$L\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $*" 2>> $O/$t.err | grep '^{' > $O/$t.jsonl; }
tll(){ local t=$1; shift; singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$L/cacheL TRITON_CACHE_DIR=$L/cacheL/triton N5_STACK=lumi; cd $L; python -I -c 'import sys,runpy; sys.path.insert(0,\"$L\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $*" 2>> $O/$t.err | grep '^{' > $O/$t.jsonl; }
tli pre_img pre_shift.py r$1 3
tll pre_lumi pre_shift.py r$1 3
tli moe_img moe_shift.py $DATA r$1 5
tll moe_lumi moe_shift.py $DATA r$1 5
date -u >> $O/time.txt
