#!/bin/bash
# Experiment A: two processes in one job, same data values, arm order shuffled per round and printed.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; mkdir -p raw/s12
for p in 1 2; do echo "{\"process\":$p,\"t\":\"$(date +%T)\"}" >> raw/s12/a_$J.jsonl; ./hbm3 s12_taba.txt 4 $((J+p)) >> raw/s12/a_$J.jsonl 2>&1; done
