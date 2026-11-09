#!/bin/bash
# N4 lever C follow-up: where the hand kernel crosses hipBLASLt between 2048^3 (0.79-0.83) and 4096^3. Blocks at 256x128 or 128x256 in the shape tag. Placed and unplaced, 1 GCD. Bytes before the floor.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j46_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
f=0
for sp in "2560 2560 2560 2560" "3072 3072 3072 3072" "3584 3584 3584 3584" "2kx8kx2k 2048 8192 2048" "8kx2kx2k 8192 2048 2048" "1kx16kx2k 1024 16384 2048" "4kx4kx1k 4096 4096 1024" "4kx4kx512 4096 4096 512"; do set -- $sp
  if [ $f = 0 ]; then ord="placed unplaced"; f=1; else ord="unplaced placed"; f=0; fi
  for m in $ord; do
    a=""; [ $m = placed ] && a="- placed"
    python ../../scripts/gemm_prefetch.py auto $2 $3 $4 3 400 40 $a > $O/auto_$1_$m.jsonl 2> $O/auto_$1_$m.err
  done
  date -u >> $O/time.txt
done
