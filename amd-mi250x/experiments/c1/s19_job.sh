#!/bin/bash
# C1 s19: allocation test. Fresh processes interleaved by mode (rotated each pass): plain hipMalloc, contiguous, one contiguous block (block on the first 4 passes only, to hold the GPU-hours). Arm order shuffled per round per process by the harness; poisoned-C checksum on first and last round.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; D=raw/s19; mkdir -p $D; O=$D/m_$J.jsonl
TAB=${TAB:-s19_tab.txt}; BIG=${BIG:-s19_big.txt}
date -u > $D/time_$J.txt; python ../N4/map23.py > $D/map_$J.txt
modes=(plain contig block)
for p in 0 1 2 3 4 5 6 7 8 9; do for i in 0 1 2; do m=${modes[$(( (i+p)%3 ))]}; [ $m = block ] && [ $p -gt 3 ] && continue; echo "{\"process\":\"$m.$p\",\"wall\":$(date +%s.%N)}" >> $O; ./hbm4 $TAB 2 $((J+100*p+i)) $m >> $O 2>&1; done; done
date -u >> $D/time_$J.txt
for p in 0 1; do for i in 0 1 2; do m=${modes[$(( (i+p)%3 ))]}; echo "{\"process\":\"big.$m.$p\",\"wall\":$(date +%s.%N)}" >> $O; ./hbm4 $BIG 2 $((J+7000+100*p+i)) $m >> $O 2>&1; done; done
date -u >> $D/time_$J.txt
if [ -n "$BEST" ]; then
  cd $R/exp/N4; export TILELANG_CACHE_DIR=$PWD/cache TRITON_CACHE_DIR=$PWD/cache/triton
  md5sum ../../scripts/gemm_prefetch.py ../../scripts/bench_gemm.py > $C/$D/best_md5_$J.txt
  python ../../scripts/bench_gemm.py --best 2 2>&1 | grep -v "INFO\|Warning" > $C/$D/best_hand_$J.jsonl
  python ../../scripts/bench_gemm.py --best 2 --compiled 2>&1 | grep -v "INFO\|Warning" > $C/$D/best_compiled_$J.jsonl
  date -u >> $C/$D/time_$J.txt
fi
