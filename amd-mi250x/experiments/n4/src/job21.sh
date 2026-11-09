#!/bin/bash
# N4 round 2, step 1: the hand kernel through the real script (python gemm_prefetch.py auto) on all 9 shapes, then the C++ harness on the same shape.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j21_$J; mkdir -p $O; date -u > $O/time.txt
for sp in "8k 8192 8192 8192 5 10 3" "6k 6144 6144 6144 5 15 3" "17k 4096 17408 5120 5 10 3" "12k 12288 12288 12288 3 4 1" "16k 16384 16384 16384 3 3 1" "4k 4096 4096 4096 5 20 3" "5632 5632 3200 4096 5 15 3" "32k 32768 1024 2048 5 20 3" "4kk 16384 4096 4096 5 10 3"; do set -- $sp; sn=$1
  python pollm.py $O/metrics_$sn.jsonl 0.1 & PP=$!
  python ../../scripts/gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 > $O/auto_$sn.jsonl 2> $O/auto_$sn.err
  n=h9_25664c8d1p32_$sn
  ./hbn $2 $3 $4 randn $5 $6 $7 0 hb $n,h9/$n.co,256,128,256,61440 > $O/hb_${sn}_randn.jsonl 2>&1
  kill $PP; wait $PP 2>/dev/null
  gzip -f $O/metrics_$sn.jsonl
done
date -u >> $O/time.txt
