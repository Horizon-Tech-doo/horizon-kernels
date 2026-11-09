#!/bin/bash
# N4/C1 round 3: repeat of the compiled kernels at offsets 0 and 128 in the same allocation (job 37 was the first read), 8192^3 and 4096x17408x5120. Arms: hb, 128x256 hand, compiled rr25632w2 and b25632a2.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j40_$J; mkdir -p $O; date -u > $O/time.txt
export HBR_DELTAS=0,128
C=../C1/s12
for sp in "8k 8192 8192 8192 30 2" "17k 4096 17408 5120 40 2"; do set -- $sp; sn=$1
  ./hbr $2 $3 $4 plain XbbXbbXbbXbb $5 $6 hb h11_o128_$sn,h11/h11_o128_$sn.co,128,256,256,61440 rr25632w2_$sn,$C/rr25632w2.asis_$sn.hsaco,256,128,256,24576 base_kernel,$C/b25632a2.asis_$sn.hsaco,256,128,256,24576 > $O/comp_$sn.jsonl 2>&1
done
date -u >> $O/time.txt
