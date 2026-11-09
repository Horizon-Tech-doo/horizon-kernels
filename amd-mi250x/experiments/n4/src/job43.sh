#!/bin/bash
# N4 lever A: compiled forms under placement, one GCD. $1 = order flip (0/1). --best compiled placed and unplaced, then select_tile's form on 9 shapes placed and unplaced. Pure TileLang (tile); rr* post-processed in --best.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j43_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt; echo "flip=$1" >> $O/time.txt
F=$1
if [ "$F" = 0 ]; then ord="placed unplaced"; else ord="unplaced placed"; fi
for m in $ord; do
  a=""; [ $m = placed ] && a="--placed"
  python ../../scripts/bench_gemm.py --best 2 --compiled $a > $O/best_$m.out 2> $O/best_$m.err
done
for sp in "4k 4096 4096 4096 600 3" "8k 8192 8192 8192 100 2" "16k 16384 16384 16384 12 1" "6k 6144 6144 6144 200 2" "12k 12288 12288 12288 30 1" "17k 4096 17408 5120 140 2" "32k 32768 1024 2048 600 3" "4kk 16384 4096 4096 200 2" "5632 5632 3200 4096 500 3"; do set -- $sp; sn=$1; MM=$2; NN=$3; KK=$4; II=$5; WW=$6
  for m in $ord; do
    a="tile"; [ $m = placed ] && a="tile placed"
    python ../../scripts/gemm_prefetch.py auto $MM $NN $KK 3 $II $WW $a > $O/tile_${sn}_$m.jsonl 2> $O/tile_${sn}_$m.err
  done
  if [ "$ord" = "placed unplaced" ]; then ord="unplaced placed"; else ord="placed unplaced"; fi
done
date -u >> $O/time.txt
