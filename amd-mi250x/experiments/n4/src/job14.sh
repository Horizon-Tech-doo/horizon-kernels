#!/bin/bash
# Headline repeat: fewer arms, more iterations, randn in two processes and fixed in one, rotated order.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j14_$J; mkdir -p $O; S=/scratch/project_XXXXXXXXX/tilelang/exp/E1/solo
python pollm.py $O/metrics.jsonl 0.05 & PP=$!
for sp in "8k 8192 8192 8192 5 20 3" "16k 16384 16384 16384 5 6 2" "4k 4096 4096 4096 5 40 3" "17k 4096 17408 5120 5 20 3" "32k 32768 1024 2048 5 40 3"; do set -- $sp; sn=$1; M=$2;N=$3;K=$4;R=$5;IT=$6;W=$7
  A="hb rr25632w2_$sn,$S/rr25632w2_$sn.co,256,128,256,24576 rr12864w3c16_$sn,sw/rr12864w3c16_$sn.co,128,128,256,32768"
  for n in h7_25664c8d1p32 h7_25664c16d1p32 h7_12825664r8d1p32; do n=${n}_$sn; A="$A $n,h7/$n.co,$(cat h7/$n.geom)"; done
  ./hbn $M $N $K randn $R $IT $W 0 $A > $O/hb_${sn}_randn1.jsonl 2>&1
  ./hbn $M $N $K fixed $R $IT $W 0 $A > $O/hb_${sn}_fixed.jsonl 2>&1
  ./hbn $M $N $K randn $R $IT $W 0 $A > $O/hb_${sn}_randn2.jsonl 2>&1
  ./hbn $M $N $K randn 3 $IT $W 5000 $A > $O/hb_${sn}_probe.jsonl 2>&1
done
kill $PP; date -u >> $O/time.txt
