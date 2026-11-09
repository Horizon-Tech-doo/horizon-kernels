#!/bin/bash
# Hand-scheduled K loop with block swizzle, against hipBLASLt and the best earlier compiled forms.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j4_$J; mkdir -p $O; S=/scratch/project_XXXXXXXXX/tilelang/exp/E1/solo
python pollm.py $O/metrics.jsonl 0.05 & PP=$!
for sp in "8k 8192 8192 8192 3 10 3" "16k 16384 16384 16384 3 3 1" "4k 4096 4096 4096 3 20 3" "17k 4096 17408 5120 3 10 3" "32k 32768 1024 2048 3 20 3"; do set -- $sp; sn=$1; M=$2;N=$3;K=$4;R=$5;IT=$6;W=$7
  A="hb rr25632w2_$sn,$S/rr25632w2_$sn.co,256,128,256,24576 b25632w2_$sn,$S/b25632w2_$sn.co,256,128,256,24576 rr12864w3c16_$sn,sw/rr12864w3c16_$sn.co,128,128,256,32768 rr12864w3c8_$sn,sw/rr12864w3c8_$sn.co,128,128,256,32768"
  for f in h7/h*_$sn.co; do n=$(basename $f .co); A="$A $n,$f,$(cat h7/$n.geom)"; done
  ./hbn $M $N $K randn $R $IT $W 0 $A > $O/hb_${sn}_randn.jsonl 2>&1
  ./hbn $M $N $K randn 1 $IT $W 5000 $A > $O/hb_${sn}_probe.jsonl 2>&1
done
kill $PP; date -u >> $O/time.txt
