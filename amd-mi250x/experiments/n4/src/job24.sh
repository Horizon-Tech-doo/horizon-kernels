#!/bin/bash
# N4 round 2, step 3b: at the package cap (all 8 GCDs busy), can another form hold 16384^3 and 6144^3 above hipBLASLt? 5 s per arm, two rotated rounds, gpu_metrics on.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j24_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
pick(){ local sn=$1; shift; local a=""; for f in "$@"; do a="$a $(tr ' ' '\n' < arms10_$sn.txt | grep "^h10_${f}_$sn,")"; done; echo $a; }
for sp in "16k 16384 16384 16384 62 2 c8 o128c8 o128r8 c4 c16 r8 p16" "6k 6144 6144 6144 1170 3 c8 c4 o128c8 o128r8 p16"; do set -- $sp; sn=$1;M=$2;N=$3;K=$4;it=$5;w=$6; shift 6; A=$(pick $sn "$@")
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbn $M $N $K randn 2 $it $w 0 hb $A > $O/all_d${d}_$sn.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
