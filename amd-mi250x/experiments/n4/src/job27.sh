#!/bin/bash
# N4 round 2: wiring check of the landed rule at the two shapes it moves to 128x256, through the real script.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j27_$J; mkdir -p $O; date -u > $O/time.txt
for sp in "6k 6144 6144 6144 5 15 3" "16k 16384 16384 16384 3 3 1"; do set -- $sp; sn=$1
  python ../../scripts/gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 > $O/auto_$sn.jsonl 2> $O/auto_$sn.err
done
date -u >> $O/time.txt
