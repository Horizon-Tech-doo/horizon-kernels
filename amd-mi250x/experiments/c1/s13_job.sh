#!/bin/bash
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s13
export TILELANG_CACHE_DIR=$C/cache_s13 TRITON_CACHE_DIR=$C/cache_s13/triton
python -I -c "import sys; sys.path.insert(0,'$C'); sys.argv=['s13_py.py','6','$J']; import runpy; runpy.run_path('$C/s13_py.py', run_name='__main__')" 2>&1 | grep -v "INFO\|Warning" >> raw/s13/b_$J.jsonl
