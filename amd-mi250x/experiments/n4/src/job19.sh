#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j19_$J; mkdir -p $O
for m in auto solo three keepc noref auto solo three keepc; do python diag.py $m >> $O/diag.jsonl 2>> $O/diag.err; done
