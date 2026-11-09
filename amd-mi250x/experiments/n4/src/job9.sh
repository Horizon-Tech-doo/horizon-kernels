#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
J=$SLURM_JOB_ID; O=raw/j9_$J; mkdir -p $O
./micro32 > $O/micro32.txt 2>&1
