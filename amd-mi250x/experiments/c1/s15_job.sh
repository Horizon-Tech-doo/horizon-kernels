#!/bin/bash
# Is placement why 8192^3 reads lower in hbm3 than in earlier jobs? Fresh processes: 8k alone, against 4k first and freed, then 8k. Same arms, same data values at 8k not guaranteed (the generator advances), so only speed is compared.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s15
ARMS='^hipblaslt_direct|b25632a2.asis|rr25632w2.asis|b12864.fill3|b12864.asis'
grep -E " 8192 8192 8192 " s12_taba.txt | grep -E "$ARMS" > s15_8k.txt
{ grep -E " 4096 4096 4096 " s12_taba.txt | grep -E "$ARMS"; cat s15_8k.txt; } > s15_48k.txt
for c in 1 2 3; do
  for t in 8k 48k; do echo "{\"process\":\"$t.$c\",\"t\":\"$(date +%T)\"}" >> raw/s15/p_$J.jsonl; ./hbm3 s15_$t.txt 3 $((J+c)) >> raw/s15/p_$J.jsonl 2>&1; done
done
