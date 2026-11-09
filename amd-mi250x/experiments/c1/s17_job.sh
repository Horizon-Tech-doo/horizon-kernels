#!/bin/bash
# B + C repeat through s13_py, same-job direct hipBLASLt from hbm3 on the same five shapes, then the landed script scripts/bench_gemm.py --best with its own cache.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s17
export TILELANG_CACHE_DIR=$C/cache_s13 TRITON_CACHE_DIR=$C/cache_s13/triton
python -I -c "import sys; sys.path.insert(0,'$C'); sys.argv=['s13_py.py','4','$J']; import runpy; runpy.run_path('$C/s13_py.py', run_name='__main__')" 2>&1 | grep -v "INFO\|Warning" >> raw/s17/b_$J.jsonl
{ grep -E "^hipblaslt|rr25632w2.asis|rr12864w3.asis|b25632a2.asis" s12_taba.txt; grep -E "^hipblaslt|b25632a2.asis" s12_tabd.txt; } > s17_tab.txt
./hbm3 s17_tab.txt 3 $J >> raw/s17/h_$J.jsonl 2>&1
export TILELANG_CACHE_DIR=$C/cache_land TRITON_CACHE_DIR=$C/cache_land/triton
(cd $R/scripts; python bench_gemm.py --best 3) 2>&1 | grep -v "INFO\|Warning" >> raw/s17/land_$J.jsonl
