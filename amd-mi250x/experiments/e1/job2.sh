#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/E1
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
mkdir -p traces2 out
J=$SLURM_JOB_ID
echo "== job $J"; date
for r in 0 1 2; do ./clk $r; done | tee out/clk_$J.jsonl
python hb.py 0 > out/hb_r0_$J.jsonl 2>out/hb_r0_$J.err
for r in 0 1 2 3 4; do
  ./gmain tab_std.txt $r >> out/std_$J.jsonl
  python hb.py $((r+1)) >> out/hb_$J.jsonl 2>>out/hb_$J.err
  ./gmain tab_geo.txt $r >> out/geo_$J.jsonl
done
for r in 0 1 2; do ./gmain tab_ck.txt $r >> out/ck_$J.jsonl; done
date
