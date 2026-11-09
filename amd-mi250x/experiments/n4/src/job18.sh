#!/bin/bash
# Held-out check: shapes never used to pick the rule or the hand kernel's panel, plus the five tuned shapes.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j18_$J; mkdir -p $O
for sp in "4k 4096 4096 4096 3 20 3" "8k 8192 8192 8192 3 10 3" "16k 16384 16384 16384 3 3 1" "17k 4096 17408 5120 3 10 3" "32k 32768 1024 2048 3 20 3" "6k 6144 6144 6144 3 15 3" "12k 12288 12288 12288 3 4 1" "5632 5632 3200 4096 3 15 3" "4kk 16384 4096 4096 3 10 3"; do set -- $sp; sn=$1
  python gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 > $O/auto_$sn.jsonl 2> $O/auto_$sn.err
  n=h9_25664c8d1p32_$sn
  ./hbn $2 $3 $4 randn $5 $6 $7 0 hb $n,h9/$n.co,256,128,256,61440 > $O/hb_${sn}_randn.jsonl 2>&1
done
date -u >> $O/time.txt
