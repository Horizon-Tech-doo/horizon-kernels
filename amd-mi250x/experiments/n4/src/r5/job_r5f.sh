#!/bin/bash
# N4 round 5 landed bytes. $1 = fwd or rev. 1 GCD: rule shapes, then small-M through gemm_small_m placed and unplaced. Hand asm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/r5f_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py verify_r5.py > $O/md5.txt
python verify_r5.py $1 all 3 0.4 > $O/v.jsonl 2> $O/v.err; date -u >> $O/time.txt
if [ "$1" = rev ]; then ord="unplaced placed"; else ord="placed unplaced"; fi
for sp in "64 5120 5120 2000 200" "32 5120 5120 2000 200" "16 17408 5120 1500 150" "1 5120 5120 3000 300" "64 17408 5120 1500 150" "64 5120 17408 1500 150"; do set -- $sp
  for m in $ord; do a=""; [ $m = placed ] && a=placed; python ../../scripts/gemm_prefetch.py smallm $1 $2 $3 3 $4 $5 $a > $O/sm_$1x$2x$3_$m.jsonl 2> $O/sm_$1x$2x$3_$m.err; done
done
date -u >> $O/time.txt
