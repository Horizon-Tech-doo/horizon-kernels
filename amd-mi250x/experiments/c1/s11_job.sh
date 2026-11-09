#!/bin/bash
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; E1=$R/exp/E1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw
for r in 0 1 2 3; do $E1/gmain $C/s11_tab.txt $r >> raw/s11_$J.jsonl 2>&1; done
