#!/bin/bash
# N4 round 6 landed bytes. $1 = fwd or rev. Real script smallm, then same-allocation draws. Hand asm.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/r6b_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum ../../scripts/gemm_prefetch.py r6/alloc_r6.py > $O/md5.txt
ORD=$1
for sp in "16 17408 5120" "32 17408 5120" "64 17408 5120" "64 5120 5120" "32 5120 5120"; do set -- $sp; python ../../scripts/gemm_prefetch.py smallm $1 $2 $3 5 2000 200 > $O/sm_$1x$2x$3.jsonl 2> $O/sm_$1x$2x$3.err; done
date -u >> $O/time.txt
python r6/alloc_r6.py $ORD > $O/alloc.jsonl 2> $O/alloc.err
date -u >> $O/time.txt
