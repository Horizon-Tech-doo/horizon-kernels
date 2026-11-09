#!/bin/bash
# N4 round 3: the 128 B A-B offset at the cap. 8 GCDs, 8 shapes, 2 fresh allocations per process; each read once with B at offset 0 and once at 128. hb, 256x128 c8, 128x256 c8, about 1.5 s per arm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/${JN:-j36}_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt; python pollm.py $O/metrics.jsonl 0.1 & PP=$!
export HBR_DELTAS=128
for sp in "16k 16384 16384 16384 18 1" "6k 6144 6144 6144 350 2" "8k 8192 8192 8192 150 2" "12k 12288 12288 12288 45 1" "4k 4096 4096 4096 1140 3" "17k 4096 17408 5120 215 2" "32k 32768 1024 2048 1080 3" "4kk 16384 4096 4096 300 2"; do set -- $sp; sn=$1
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbr $2 $3 $4 plain XbXb $5 $6 hb h11_c8_$sn,h11/h11_c8_$sn.co,256,128,256,61440 h11_o128_$sn,h11/h11_o128_$sn.co,128,256,256,61440 > $O/cap_${sn}_d$d.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
