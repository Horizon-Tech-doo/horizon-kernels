#!/bin/bash
# N4 lever B: script with final bytes, placed, 8 GCDs at the cap: 8192^3 first, then 16384^3, 6144^3, 12288^3.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j44_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
for sp in "8k 8192 2 100 2" "16k 16384 2 12 1" "6k 6144 2 233 2" "12k 12288 2 30 1"; do set -- $sp; sn=$1
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d python ../../scripts/gemm_prefetch.py auto $2 $2 $2 $3 $4 $5 - placed > $O/script_${sn}_d$d.jsonl 2> $O/script_${sn}_d$d.err & pids+=($!); done
  wait "${pids[@]}"; date -u >> $O/time.txt
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
