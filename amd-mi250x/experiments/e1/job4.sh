#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/E1
J=$SLURM_JOB_ID
./hbm2 3 > out/hbm2_$J.jsonl 2>&1
date
