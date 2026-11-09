#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:09:00
#SBATCH --job-name=n5r2land
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out/slurm-%j.out
# Landed module V2 from empty caches: land_test on both stacks (fast then plain in one process, the order that hit the stale bug), real script on the image stack.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
date
for st in I L; do
  rm -rf $E/cache/v2$st
  echo "=== land_test $st $(date +%T)"
  timeout 200 bash -c "source $S/n5env.sh; N5_ENV='TILELANG_CACHE_DIR=$E/cache/v2$st'; TL_$st '' 'land_test.py'" 2>>err.log | grep '^{' | sed -u "s/^{/{\"stack\": \"$st\", /" | tee -a land.jsonl
done
for a in "4096 - 128" "8192 - 256"; do
  echo "=== main I $a $(date +%T)"
  timeout 90 bash -c "source $S/n5env.sh; N5_ENV='TILELANG_CACHE_DIR=$E/cache/v2I'; TL_I '' 'gqa_fa.py $a'" 2>>err.log | grep '^{' | tee -a main.jsonl
done
rm -rf $E/cache/v2I $E/cache/v2L
date; echo DONE
