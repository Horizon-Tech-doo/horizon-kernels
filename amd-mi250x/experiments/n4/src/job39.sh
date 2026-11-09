#!/bin/bash
# N4 round 3: final script bytes. bench_gemm --best with place_b, and place_b's cost.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j39_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
python ../../scripts/bench_gemm.py --best 2 --placed 2>&1 | grep -v "INFO\|Warning" > $O/best_placed.jsonl
python placecost.py > $O/placecost.jsonl 2>&1
date -u >> $O/time.txt
