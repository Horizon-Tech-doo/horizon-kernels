#!/bin/bash
# N4 round 3, step 0: the final script bytes on all 9 shapes, solo. Then the 32-bit boundary: largest accepted hand shape, and two shapes the guard sends to the tile form. Then a smoke run of the C1 s19 harness.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/j28_$J; mkdir -p $O; date -u > $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $O/md5.txt
python map23.py > $O/map.txt; python pollm.py $O/metrics.jsonl 0.2 & PP=$!
for sp in "8k 8192 8192 8192 3 10 3" "6k 6144 6144 6144 3 15 3" "17k 4096 17408 5120 3 10 3" "12k 12288 12288 12288 3 4 1" "16k 16384 16384 16384 3 3 1" "4k 4096 4096 4096 3 20 3" "5632 5632 3200 4096 3 15 3" "32k 32768 1024 2048 3 20 3" "4kk 16384 4096 4096 3 10 3" "b46k 46336 46336 46336 1 1 0" "b46k8 46592 46336 8192 1 1 0" "b65 65536 1024 65536 1 1 0"; do set -- $sp; sn=$1
  python ../../scripts/gemm_prefetch.py auto $2 $3 $4 $5 $6 $7 > $O/auto_$sn.jsonl 2> $O/auto_$sn.err
done
kill $PP; wait $PP 2>/dev/null; gzip -f $O/metrics.jsonl; date -u >> $O/time.txt
cd ../C1; D=raw/s19; mkdir -p $D
for m in plain contig block; do echo "{\"process\":\"smoke.$m\"}" >> $D/smoke_$J.jsonl; ./hbm4 s19_smoke.txt 1 $J $m >> $D/smoke_$J.jsonl 2>&1; echo "rc $?" >> $D/smoke_$J.jsonl; done
cd ../N4; date -u >> $O/time.txt
