#!/bin/bash
# N4 round 3, step 3: eight shapes at the cap on a node other than the earlier cap node (excluded at submit). 3 s per arm, 2 rotated rounds. First: hb kernel names at 6144^3 and 12288^3.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j31_$J; mkdir -p $O; date -u > $O/time.txt
python map23.py > $O/map.txt
for sp in "6k 6144" "12k 12288"; do set -- $sp
  ROCR_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=4 ./hbn $2 $2 $2 randn 1 1 0 0 hb 2>&1 >/dev/null | grep -aE "ShaderName|grid=|group_seg" | grep -aE "Cijk|grid=|group_seg" | sed -E "s/^.*(ShaderName)/\1/; s/^.*(grid=)/\1/" | cut -c1-260 | uniq -c > $O/hblog_direct_$1.txt
done
date -u >> $O/time.txt
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
for sp in "4k 4096 4096 4096 2280 3" "12k 12288 12288 12288 90 2" "17k 4096 17408 5120 430 3" "32k 32768 1024 2048 2160 3" "5632 5632 3200 4096 2100 3" "4kk 16384 4096 4096 600 3" "16k 16384 16384 16384 37 2" "6k 6144 6144 6144 700 3"; do set -- $sp; sn=$1;M=$2;N=$3;K=$4;it=$5;w=$6
  a="hb h11_c8_$sn,h11/h11_c8_$sn.co,256,128,256,61440"; [ -f h11/h11_o128_$sn.co ] && a="$a h11_o128_$sn,h11/h11_o128_$sn.co,128,256,256,61440"
  pids=(); for d in 0 1 2 3 4 5 6 7; do ROCR_VISIBLE_DEVICES=$d ./hbn $M $N $K randn 2 $it $w 0 $a > $O/all_d${d}_$sn.jsonl 2>&1 & pids+=($!); done
  wait "${pids[@]}"
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
