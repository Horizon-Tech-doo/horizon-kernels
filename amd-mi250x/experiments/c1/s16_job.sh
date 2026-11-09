#!/bin/bash
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s16
for p in 1 2; do echo "{\"process\":$p,\"t\":\"$(date +%T)\"}" >> raw/s16/a22_$J.jsonl; ./hbm3_22 s16_tab.txt 4 $((J+p)) >> raw/s16/a22_$J.jsonl 2>&1; done
