#!/bin/bash
# Default main of the landed script, plain and fast in one process.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
for st in L I; do
  for a in "4096 - 128" "8192 - 256" "1024 - 128" "16384 - 256"; do
    bash -c "source $S/n5env.sh; N5_ENV='TILELANG_CACHE_DIR=$E/cache/land$st'; TL_$st '' 'gqa_fa.py $a'" 2>/dev/null | grep '^{' | sed -u "s/^{/{\"stack\": \"$st\", /" | tee -a main2_$st.jsonl
  done
done
date; echo DONE
