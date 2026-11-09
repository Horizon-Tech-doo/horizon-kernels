#!/bin/bash
# SMALLM. $1 = fwd or rev. All arms hand asm; hipBLASLt through torch same process, direct after for reference.
cd /scratch/project_XXXXXXXXX/tilelang/exp/SMALLM
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/sm7_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum ../../scripts/gemm_prefetch.py sm7.py > $O/md5.txt
python sm7.py $1 > $O/sm7.jsonl 2> $O/sm7.err; echo "rc=$?" >> $O/time.txt; date -u >> $O/time.txt
export HBR_DELTAS=0
while read -r s M N K it w; do ../N4/hbr $M $N $K plain bbbb $it $w hb > $O/d_$s.jsonl 2>&1; done <<L
16 16 17408 5120 3000 300
64 64 5120 5120 3000 300
L
date -u >> $O/time.txt
