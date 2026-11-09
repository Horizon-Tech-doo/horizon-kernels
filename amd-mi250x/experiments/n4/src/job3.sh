#!/bin/bash
# Swizzle and K-start stagger race. Same process as hipBLASLt, randn, rotated, UMC read from gpu_metrics.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j3_$J; mkdir -p $O; S=/scratch/project_XXXXXXXXX/tilelang/exp/E1/solo
python pollm.py $O/metrics.jsonl 0.05 & PP=$!
date -u > $O/time.txt
arms(){ sn=$1; echo hb
 for b in "rr25632w2 256 128 24576" "b25632w2 256 128 24576" "rr12864w3 128 128 32768"; do set -- $b; echo $1"_$sn,$S/$1""_$sn.co,$2,$3,256,$4"; for t in s4 r4 r8 r16 c8 c16 r4s4 r8s4 r16s4 c8s4 c16s4; do echo $1$t"_$sn,sw/$1$t""_$sn.co,$2,$3,256,$4"; done; done; }
for sp in "16k 16384 16384 16384 3 3 1" "8k 8192 8192 8192 3 10 3" "4k 4096 4096 4096 3 20 3" "17k 4096 17408 5120 3 10 3" "32k 32768 1024 2048 3 20 3"; do set -- $sp; sn=$1; M=$2;N=$3;K=$4;R=$5;IT=$6;W=$7
  echo "SHAPE $sn $(date -u +%T)" >> $O/time.txt
  ./hbn $M $N $K randn $R $IT $W 0 $(arms $sn) > $O/hb_${sn}_randn.jsonl 2>&1
done
kill $PP; date -u >> $O/time.txt
