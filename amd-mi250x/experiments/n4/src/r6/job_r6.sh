#!/bin/bash
# N4 round 6. $1 = fwd or rev. Padded-N tiles against the current ones through python, hb through torch same process, then direct hb same job. Hand asm except S1.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/r6a_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum ../../scripts/gemm_prefetch.py r6/pad_r6.py hbr > $O/md5.txt
python r6/pad_r6.py $1 3 > $O/pad.jsonl 2> $O/pad.err; date -u >> $O/time.txt
export HBR_DELTAS=0,128
while read -r s M N K it w; do ./hbr $M $N $K plain bbbb $it $w hb > $O/d_$s.jsonl 2>&1; done <<L
16w 16 17408 5120 3000 300
64w 64 17408 5120 3000 300
32w 32 17408 5120 3000 300
7168 7168 3584 4096 240 12
3072 3072 3072 3072 800 40
L
date -u >> $O/time.txt
