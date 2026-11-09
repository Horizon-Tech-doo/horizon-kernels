#!/bin/bash
# N4 round 2, step 2: forms, LDS base offset, independent strides, guarded last load. One process per shape, hb in the process, arms rotated by the harness.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j22_$J; mkdir -p $O; date -u > $O/time.txt
python pollm.py $O/metrics.jsonl 0.2 & PP=$!
for sp in "6k 6144 6144 6144 7 30 5" "12k 12288 12288 12288 7 10 2" "16k 16384 16384 16384 7 5 1" "32k 32768 1024 2048 9 30 5"; do set -- $sp; sn=$1
  ./hbn $2 $3 $4 randn $5 $6 $7 0 hb $(cat arms10_$sn.txt) > $O/hb_$sn.jsonl 2>&1
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
