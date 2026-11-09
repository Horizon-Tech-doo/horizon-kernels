#!/bin/bash
# N4 after the second floor: short-side shapes must return rule None; then the solo pass on the final bytes, 9 shapes, placed.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j48_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
for sp in "256x32k 256 32768 2048" "32kx256 32768 256 2048"; do set -- $sp
  for m in unplaced placed; do a=""; [ $m = placed ] && a="- placed"; python ../../scripts/gemm_prefetch.py auto $2 $3 $4 1 10 1 $a > $O/none_$1_$m.jsonl 2> $O/none_$1_$m.err; done
done
for sp in "4k 4096 4096 4096 600 3" "8k 8192 8192 8192 100 2" "16k 16384 16384 16384 12 1" "6k 6144 6144 6144 200 2" "12k 12288 12288 12288 30 1" "17k 4096 17408 5120 140 2" "32k 32768 1024 2048 600 3" "4kk 16384 4096 4096 200 2" "5632 5632 3200 4096 500 3"; do set -- $sp
  python ../../scripts/gemm_prefetch.py auto $2 $3 $4 3 $5 $6 - placed > $O/auto_$1.jsonl 2> $O/auto_$1.err
done
date -u >> $O/time.txt
