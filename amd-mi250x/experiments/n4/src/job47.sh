#!/bin/bash
# N4 lever C after the floor: below-floor shapes must return rule None; above-floor extremes (aspect, K) and 3072^3 repeated, placed and unplaced, 1 GCD.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j47_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
for sp in "s128 128 256 192" "s512 512 1024 512" "s1k 1024 1024 1024" "s2k 2048 2048 2048" "m128 128 5120 5120"; do set -- $sp
  for m in unplaced placed; do a=""; [ $m = placed ] && a="- placed"; python ../../scripts/gemm_prefetch.py auto $2 $3 $4 1 10 1 $a > $O/none_$1_$m.jsonl 2> $O/none_$1_$m.err; done
done
f=0
for sp in "2560 2560 2560 2560" "3072 3072 3072 3072" "256x32k 256 32768 2048" "32kx256 32768 256 2048" "2560k256 2560 2560 256" "4kk192 4096 4096 192"; do set -- $sp
  if [ $f = 0 ]; then ord="placed unplaced"; f=1; else ord="unplaced placed"; f=0; fi
  for m in $ord; do a=""; [ $m = placed ] && a="- placed"; python ../../scripts/gemm_prefetch.py auto $2 $3 $4 3 400 40 $a > $O/auto_$1_$m.jsonl 2> $O/auto_$1_$m.err; done
  date -u >> $O/time.txt
done
