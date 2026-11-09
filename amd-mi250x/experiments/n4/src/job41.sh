#!/bin/bash
# N4 round 3: final script bytes, rule without placement, at the 4 shapes of HAND_128X256 and at 8192^3. One GCD.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j41_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
for sp in "6k 6144 6144 6144 3 15 3" "16k 16384 16384 16384 3 3 1" "12k 12288 12288 12288 3 4 1" "17k 4096 17408 5120 3 10 3" "8k 8192 8192 8192 3 10 3"; do set -- $sp
  python ../../scripts/gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 > $O/auto_$1.jsonl 2> $O/auto_$1.err
done
date -u >> $O/time.txt
