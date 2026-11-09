#!/bin/bash
# N4 round 5 job B. $1 = fwd or rev. Rule gap through the unchanged LUMI script (hand asm), placed and unplaced in separate processes, torch hb in the same process. Then small-M (compiled), torch hb, plus direct hb and its kernel names.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4
export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
J=$SLURM_JOB_ID; O=raw/r5b_$J; mkdir -p $O; date -u > $O/time.txt; echo "order=$1" >> $O/time.txt; md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py sm_r5.py hbn > $O/md5.txt
if [ "$1" = rev ]; then ord="unplaced placed"; else ord="placed unplaced"; fi
for sp in "1kx8k 1024 8192 2048" "8kx1k 8192 1024 2048" "2kx4k 2048 4096 2048" "4kx2k 4096 2048 2048" "1kx6656 1024 6656 2048" "6656x1k 6656 1024 2048" "1536x4608 1536 4608 2048"; do set -- $sp
  for m in $ord; do a=""; [ $m = placed ] && a="- placed"; python ../../scripts/gemm_prefetch.py auto $2 $3 $4 3 1200 120 $a > $O/gap_$1_$m.jsonl 2> $O/gap_$1_$m.err; done
  date -u >> $O/time.txt
done
python sm_r5.py $1 3 > $O/sm.jsonl 2> $O/sm.err; date -u >> $O/time.txt
for sp in "64x5k 64 5120 5120 800 80" "32x5k 32 5120 5120 800 80" "16x17k 16 17408 5120 800 80" "64x17k 64 17408 5120 800 80" "64x5kK17k 64 5120 17408 800 80" "1x5k 1 5120 5120 2000 200"; do set -- $sp
  ./hbn $2 $3 $4 randn 3 $5 $6 0 hb > $O/hbdirect_$1.jsonl 2>&1
  ROCR_VISIBLE_DEVICES=0 AMD_LOG_LEVEL=4 ./hbn $2 $3 $4 randn 1 1 0 0 hb 2>&1 >/dev/null | grep -aE "ShaderName|grid=" | grep -aE "Cijk|grid=" | sed -E "s/^.*(ShaderName)/\1/; s/^.*(grid=)/\1/; s/kernel_obj.*//" | cut -c1-200 | uniq -c > $O/hblog_$1.txt
done
date -u >> $O/time.txt
