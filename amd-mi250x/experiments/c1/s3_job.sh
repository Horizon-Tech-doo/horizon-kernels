#!/bin/bash
# GEMM: JIT routes vs hand-compiled E1 kernels vs hipBLASLt in one allocation. Arm order rotates per round.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; E1=$R/exp/E1; cd $C
J=$SLURM_JOB_ID; O=raw/s3_$J; mkdir -p $O s3co
echo "job $J order per round recorded below"; rocm-smi --showclocks 2>/dev/null | grep -m1 sclk
run_wheel(){ TILELANG_CACHE_DIR=$C/cache_wheel python s3_py.py wheel $1 2>&1 | grep -E '^\{|Error|error' >> $O/py.jsonl; }
run_hook(){ TILELANG_CACHE_DIR=$C/cache_hook python s3_py.py hook $1 2>&1 | grep -E '^\{|Error|error' >> $O/py.jsonl; }
run_fork(){ PYTHONPATH=$R/fork LD_LIBRARY_PATH=$R/fork/build/lib:$LD_LIBRARY_PATH TILELANG_CACHE_DIR=$C/cache_fork python s3_py.py fork $1 2>&1 | grep -E '^\{|Error|error' >> $O/py.jsonl; }
run_gmain(){ $E1/gmain $C/s3_tab.txt $1 2>&1 >> $O/gmain.jsonl; }
run_hbm(){ (cd $E1; ./hbm2 1) >> $O/hbm2.jsonl 2>&1; }
ORDERS=("wheel hook fork gmain hbm" "fork gmain hbm wheel hook" "hbm hook wheel fork gmain")
for r in 0 1 2; do
 echo "round $r: ${ORDERS[$r]}"; for a in ${ORDERS[$r]}; do date +%T; run_$a $r; 
  if [ $r = 0 ] && [ $a = fork ]; then :; fi; done
done
rocm-smi --showclocks 2>/dev/null | grep -m1 sclk; date +%T
