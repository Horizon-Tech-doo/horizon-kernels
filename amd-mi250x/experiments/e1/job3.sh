#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/E1
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID
AMD_LOG_LEVEL=3 python hb_log.py > out/hblog_$J.out 2> out/hblog_$J.raw
grep -E "===SHAPE|ShaderName|grid=|Dispatch Header|group_seg" out/hblog_$J.raw | grep -vE "shader_" | sed -E 's/^.*(ShaderName)/\1/' | cut -c1-400 > out/hblog_$J.txt
wc -l out/hblog_$J.txt; rm -f out/hblog_$J.raw
python bf.py > out/bf_$J.jsonl 2> out/bf_$J.err
date
./hbm 5 > out/hbm_$J.jsonl 2>&1
AMD_LOG_LEVEL=3 ./hbm 1 2> out/hbmlog_$J.raw > /dev/null; grep -E "ShaderName" out/hbmlog_$J.raw | sed -E 's/^.*(ShaderName)/\1/' | sort | uniq -c | cut -c1-250 > out/hbmlog_$J.txt; rm -f out/hbmlog_$J.raw
date
