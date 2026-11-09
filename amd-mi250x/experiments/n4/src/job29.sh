#!/bin/bash
# N4 round 3, step 1: 8 GCDs, 6 reallocations per process, hb and hand in each, metrics on.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j29_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt; python pollm.py $O/metrics.jsonl 0.1 & PP=$!
pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d python n4s1.py 6 16 > $O/s1_d$d.jsonl 2> $O/s1_d$d.err & pids+=($!); done
wait "${pids[@]}"
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
