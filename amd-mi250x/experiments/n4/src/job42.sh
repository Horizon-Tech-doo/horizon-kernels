#!/bin/bash
# N4 lever C: small shapes, script through auto, placed and unplaced, 1 GCD. Reads rel_err and ratio. Order placed/unplaced flips per shape.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j42_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
f=0
for sp in "s128 128 256 192 3 2000 200" "s256 256 256 256 3 1500 200" "s512 512 1024 512 3 1500 150" "s1k 1024 1024 1024 3 1500 150" "s2k 2048 2048 2048 3 500 50" "m128 128 5120 5120 3 1500 150"; do set -- $sp
  if [ $f = 0 ]; then ord="placed unplaced"; f=1; else ord="unplaced placed"; f=0; fi
  for m in $ord; do
    if [ $m = placed ]; then a="- placed"; else a=""; fi
    python ../../scripts/gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 $a > $O/auto_$1_$m.jsonl 2> $O/auto_$1_$m.err
  done
done
date -u >> $O/time.txt
