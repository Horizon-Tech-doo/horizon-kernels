#!/bin/bash
# Skeptic rerun: burst vs sustained, quiet vs busy second GPU. No build. Existing h9 code objects and hbn only.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/sk_$J; mkdir -p $O
IFS=, read a b <<< "$ROCR_VISIBLE_DEVICES"
echo "rocr=$ROCR_VISIBLE_DEVICES a=$a b=$b" > $O/meta.txt; date -u >> $O/meta.txt
python pollm.py $O/metrics.jsonl 0.2 & PP=$!
run(){ # tag dev sn M N K rounds it warm
  n=h9_25664c8d1p32_$3
  ROCR_VISIBLE_DEVICES=$2 ./hbn $4 $5 $6 randn $7 $8 $9 0 hb $n,h9/$n.co,256,128,256,61440 > $O/$1_$3.jsonl 2>&1
}
burst(){ tag=$1; shift; for dev in "$@"; do (
  run ${tag}_d$dev $dev 8k 8192 8192 8192 5 10 3; run ${tag}_d$dev $dev 16k 16384 16384 16384 5 6 2; run ${tag}_d$dev $dev 6k 6144 6144 6144 5 15 3 ) & done; wait; }
sust(){ tag=$1; shift; for dev in "$@"; do (
  run ${tag}_d$dev $dev 8k 8192 8192 8192 3 800 3; run ${tag}_d$dev $dev 16k 16384 16384 16384 3 105 2; run ${tag}_d$dev $dev 6k 6144 6144 6144 3 1900 3 ) & done; wait; }
burst b1solo $a
sust  s1solo $a
burst b2both $a $b
sust  s2both $a $b
burst b3solo $a
kill $PP; date -u >> $O/meta.txt
