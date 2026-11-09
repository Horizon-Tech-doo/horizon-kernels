#!/bin/bash
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
N5_ENV=''
for st in L I; do
  echo "== $st"; bash -c "source $S/n5env.sh; N5_ENV='TILELANG_CACHE_DIR=$E/cache/land$st'; TL_$st '' 'land_test.py'" 2>/dev/null | grep '^{' | tee -a land_$st.jsonl
  bash -c "source $S/n5env.sh; N5_ENV='TILELANG_CACHE_DIR=$E/cache/land$st'; TL_$st '' 'gqa_fa.py 8192 [32,32,768,true,2,true,true,true,true,true] 256'" 2>/dev/null | grep '^{' | tee -a main_$st.jsonl
  bash -c "source $S/n5env.sh; N5_ENV='TILELANG_CACHE_DIR=$E/cache/land$st'; TL_$st '' 'gqa_fa.py 4096 [0,32,512,true,2,true,true,true,true,true,128] 128'" 2>/dev/null | grep '^{' | tee -a main_$st.jsonl
done
date; echo DONE
