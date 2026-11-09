#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=n5ab
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out4/slurm-%j.out
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets4.sh
O=$E/out4/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py $S/n5_r4.py $S/n5lib_r4.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
# A off-grid and B MHA rule hole and batch, image stack, 2 rounds, a separate process per plan
N5_ENV=""
for r in 1 2; do
  echo "=== r$r offgrid $(date +%T)"; TL_I r4 "n5_r4.py --plan plans4/offgrid.json --tag r${r}_og --check --ck" 2>> err.log | grep '^{' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
  echo "=== r$r mhab $(date +%T)"; TL_I r4 "n5_r4.py --plan plans4/mhab.json --tag r${r}_mh --check --ck --inputs randn0,big,ar" 2>> err.log | grep '^{' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
  echo "=== r$r batch $(date +%T)"; TL_I r4 "n5_r4.py --plan plans4/batch.json --tag r${r}_ba --check --ck" 2>> err.log | grep '^{' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
done
date +%T; echo DONE
