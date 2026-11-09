#!/bin/bash
# N4 round 2: orientation check on one GPU. hb, 256x128 c8, 128x256 c8, sustained 5 s per arm, two rotated rounds, every shape that takes both.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j25_$J; mkdir -p $O; date -u > $O/time.txt
c8(){ if [ -f h10/h10_c8_$1.co ]; then echo h10_c8_$1,h10/h10_c8_$1.co,256,128,256,61440; else echo h9_25664c8d1p32_$1,h9/h9_25664c8d1p32_$1.co,256,128,256,61440; fi; }
python pollm.py $O/metrics.jsonl 0.1 & PP=$!
for sp in "8k 8192 8192 8192 500 3" "16k 16384 16384 16384 62 2" "4k 4096 4096 4096 3800 3" "17k 4096 17408 5120 720 3" "6k 6144 6144 6144 1170 3" "32k 32768 1024 2048 3600 3" "12k 12288 12288 12288 150 2" "4kk 16384 4096 4096 1000 3"; do set -- $sp; sn=$1
  ./hbn $2 $3 $4 randn 2 $5 $6 0 hb $(c8 $sn) h10_o128c8_$sn,h10/h10_o128c8_$sn.co,128,256,256,61440 > $O/hb_$sn.jsonl 2>&1
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
