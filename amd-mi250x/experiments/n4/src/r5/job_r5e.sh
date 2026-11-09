#!/bin/bash
# N4 round 5 job E. $1 = fwd or rev. hbr, hb direct, same process, B offsets 0 and 128. Hand-scheduled asm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/r5e_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum gp_r5.py gen_r5e.py hbr r5/liste.txt > $O/md5.txt
export HBR_DELTAS=0,128
while read -r s M N K it w arms; do
  if [ "$1" = rev ]; then a="$(echo $arms | tr ' ' '\n' | tac | tr '\n' ' ') hb"; else a="hb $arms"; fi
  ./hbr $M $N $K plain bbbb $it $w $a > $O/e_$s.jsonl 2>&1
done < r5/liste.txt
date -u >> $O/time.txt
