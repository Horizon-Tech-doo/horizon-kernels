#!/bin/bash
# Real script sweep: nothing hand-compiled. Private TileLang cache.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j17_$J; mkdir -p $O
for sp in "8k 8192 8192 8192 3 10 3" "16k 16384 16384 16384 3 3 1" "4k 4096 4096 4096 3 20 3" "17k 4096 17408 5120 3 10 3" "32k 32768 1024 2048 3 20 3"; do set -- $sp
  for p in 1 2; do python gemm_prefetch.py sweep $2 $3 $4 $5 $6 $7 > $O/sweep_$1_$p.jsonl 2> $O/sweep_$1_$p.err; done
done
date -u >> $O/time.txt
