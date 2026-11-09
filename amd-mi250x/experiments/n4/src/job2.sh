#!/bin/bash
# K2 race. Each shape: one process per pass, arm order rotated, hipBLASLt in the same process. randn data. Second process with clock probe.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j2_$J; mkdir -p $O; S=/scratch/project_XXXXXXXXX/tilelang/exp/E1/solo; H=hand; L=solo
python pollm.py $O/metrics.jsonl 0.05 & PP=$!
date -u > $O/time.txt
arms(){ sn=$1; echo hb \
 rr25632w2_$sn,$S/rr25632w2_$sn.co,256,128,256,24576 b25632w2_$sn,$S/b25632w2_$sn.co,256,128,256,24576 rr12864w3_$sn,$S/rr12864w3_$sn.co,128,128,256,32768
 for v in rr25664 rr25664w2 rr25664w3 b25664 b25664w2 rr12825664 rr12825664w2 rr12825664w3 b12825664 b12825664w2; do case $v in *12825664*) m=128;n=256;;*) m=256;n=128;;esac; echo $v"_$sn,$L/$v""_$sn.co,$m,$n,256,49152"; done
 for v in rr12825632 rr12825632w2 rr12825632w3; do echo $v"_$sn,$L/$v""_$sn.co,128,256,256,24576"; done
 for v in b25632t512 b25632t512w2 rr25632t512 rr25632t512w2 rr25632t512w3; do echo $v"_$sn,$L/$v""_$sn.co,256,128,512,24576"; done
 for v in h1_25664 h1_25664w2 h5_25664 h5_25664w2 h3_25664 h0_25664 h1_25664s4 h1_25664s4w2; do echo $v"_$sn,$H/$v""_$sn.co,256,128,256,49152"; done
 for v in h1_12825664 h1_12825664w2 h5_12825664; do echo $v"_$sn,$H/$v""_$sn.co,128,256,256,49152"; done
 for v in h1_25632 h1_25632w2 h5_25632 h5_25632w2 h1_25632s4w2; do echo $v"_$sn,$H/$v""_$sn.co,256,128,256,24576"; done
 echo h4_25632_$sn,$H/h4_25632_$sn.co,256,128,256,49152
}
for sp in "8k 8192 8192 8192 3 10 3" "16k 16384 16384 16384 3 3 1" "4k 4096 4096 4096 3 20 3" "17k 4096 17408 5120 3 10 3" "32k 32768 1024 2048 3 20 3"; do set -- $sp; sn=$1; M=$2;N=$3;K=$4;R=$5;IT=$6;W=$7
  echo "SHAPE $sn $(date -u +%T)" >> $O/time.txt
  ./hbn $M $N $K randn $R $IT $W 0 $(arms $sn) > $O/hb_${sn}_randn.jsonl 2>&1
  ./hbn $M $N $K randn 1 $IT $W 5000 $(arms $sn) > $O/hb_${sn}_probe.jsonl 2>&1
done
kill $PP; date -u >> $O/time.txt
