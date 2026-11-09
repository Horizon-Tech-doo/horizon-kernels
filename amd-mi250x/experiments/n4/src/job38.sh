#!/bin/bash
# N4 round 3: repeat of job 36 with no node exclusion, about 1 s per arm, then the placed rule through the real script on 8 GCDs at 16384^3, 6144^3, 8192^3.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j38_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
python map23.py > $O/map.txt; python pollm.py $O/metrics.jsonl 0.1 & PP=$!
export HBR_DELTAS=128
for sp in "16k 16384 16384 16384 12 1" "6k 6144 6144 6144 233 2" "8k 8192 8192 8192 100 2" "12k 12288 12288 12288 30 1" "4k 4096 4096 4096 760 3" "17k 4096 17408 5120 143 2" "32k 32768 1024 2048 720 3" "4kk 16384 4096 4096 200 2"; do set -- $sp; sn=$1
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbr $2 $3 $4 plain XbXb $5 $6 hb h11_c8_$sn,h11/h11_c8_$sn.co,256,128,256,61440 h11_o128_$sn,h11/h11_o128_$sn.co,128,256,256,61440 > $O/cap_${sn}_d$d.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"
done
date -u >> $O/time.txt
for sp in "16k 16384 2 12 1" "6k 6144 2 233 2" "8k 8192 2 100 2"; do set -- $sp; sn=$1
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d python ../../scripts/gemm_prefetch.py auto $2 $2 $2 $3 $4 $5 - placed > $O/script_${sn}_d$d.jsonl 2> $O/script_${sn}_d$d.err & pids+=($!); done
  wait "${pids[@]}"
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
