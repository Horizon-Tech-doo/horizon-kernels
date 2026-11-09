#!/bin/bash
# N4 round 2: repeat of the cap result. All 8 GCDs, hb, 256x128 c8, 128x256 c8, 5 s per arm, two rotated rounds.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j26_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
for sp in "16k 16384 16384 16384 62 2" "6k 6144 6144 6144 1170 3" "8k 8192 8192 8192 500 3"; do set -- $sp; sn=$1;M=$2;N=$3;K=$4;it=$5;w=$6
  c=h9_25664c8d1p32_$sn,h9/h9_25664c8d1p32_$sn.co,256,128,256,61440
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbn $M $N $K randn 2 $it $w 0 hb $c h10_o128c8_$sn,h10/h10_o128c8_$sn.co,128,256,256,61440 > $O/all_d${d}_$sn.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
