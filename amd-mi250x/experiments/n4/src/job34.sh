#!/bin/bash
# N4 round 3: one GCD. Shift B against A inside one allocation, every shape: is the state a function of the A-B address offset, is the best offset a constant or tied to the row stride, does hb care? Arms hb, 256x128 c8, 128x256 c8. HBR_DELTAS lists the offsets; X = fresh allocation.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j34_$J; mkdir -p $O; date -u > $O/time.txt
export HBR_DELTAS=0,64,128,256,512,1024,2048,4096,8192,12288,16384,24576,32768,49152,65536,131072,262144,524288,1048576
s=bbbbbbbbbbbbbbbbbbb
for sp in "16k 16384 16384 16384 6 1" "6k 6144 6144 6144 60 2" "8k 8192 8192 8192 30 2" "12k 12288 12288 12288 8 1" "4k 4096 4096 4096 200 3" "17k 4096 17408 5120 40 2" "32k 32768 1024 2048 180 3" "4kk 16384 4096 4096 50 2"; do set -- $sp; sn=$1
  ./hbr $2 $3 $4 plain X${s}X${s}X${s} $5 $6 hb h11_c8_$sn,h11/h11_c8_$sn.co,256,128,256,61440 h11_o128_$sn,h11/h11_o128_$sn.co,128,256,256,61440 > $O/sh_$sn.jsonl 2>&1
done
date -u >> $O/time.txt
