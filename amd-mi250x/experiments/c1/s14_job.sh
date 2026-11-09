#!/bin/bash
# Experiment D: 8 fresh processes at 16384^3, each with direct hipBLASLt and b25632a2 and b12864 on one set of buffers. Addresses logged by hbm3.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s14
for p in 1 2 3 4 5 6 7 8; do echo "{\"process\":$p,\"t\":\"$(date +%T)\"}" >> raw/s14/d_$J.jsonl; ./hbm3 s12_tabd.txt 3 $((J+p)) >> raw/s14/d_$J.jsonl 2>&1; done
