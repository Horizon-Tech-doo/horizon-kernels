#!/bin/bash
# N4 round 5 landed bytes at the cap: 8 GCDs, one process per GCD, about 1 s per arm, placed and unplaced. Hand asm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/r5g_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py verify_r5.py > $O/md5.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d python verify_r5.py fwd cap 2 1.0 > $O/cap_d$d.jsonl 2> $O/cap_d$d.err & pids+=($!); done
wait "${pids[@]}"; kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
