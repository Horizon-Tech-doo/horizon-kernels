#!/bin/bash
# Experiment A repeat in one process, for a job that does not run beside its own twin.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s18
grep -vE "rr25632w2.hand|b12864.rep[37]" s12_taba.txt > s18_tab.txt
echo '{"process":1}' >> raw/s18/a_$J.jsonl; ./hbm3 s18_tab.txt 6 $J >> raw/s18/a_$J.jsonl 2>&1
