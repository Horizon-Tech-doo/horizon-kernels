#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=n5rx
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out3/slurm-%j.out
# stride with and without TileLang's xor, alternating processes
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out3/$SLURM_JOB_ID; mkdir -p $O; cd $O
date; md5sum $S/gqa_fa.py $S/n5_r3.py
for r in 1 2 3; do
  for m in x n n x; do
    if [ $m = x ]; then N5_ENV='N5_NOX=1'; fl=r3x; else N5_ENV=''; fl=r3; fi
    echo "=== r$r $m $(date +%T)"
    TL_I $fl "n5_r3.py --plan plans3/nox.json --tag r$r$m --check --ck --rep 30" > tmp.out 2>> err.log; echo "rc=$?"
    grep '^{' tmp.out >> run.jsonl
  done
done
rm -f tmp.out; date; echo DONE
