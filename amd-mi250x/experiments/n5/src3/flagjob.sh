#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --job-name=n5fl
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out3/slurm-%j.out
# Image stack: each flag set has its own cache, compiled on the login node. Processes rotate. No compile in the job.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets.sh
O=$E/out3/$SLURM_JOB_ID; mkdir -p $O; cd $O
date; md5sum $S/gqa_fa.py $S/n5_r3.py $S/flagsets.sh
for r in 1 2 3; do
  for i in 0 1 2 3 4; do
    k=${ORDER[$(( (i + r - 1) % 5 ))]}
    echo "=== r$r $k $(date +%T)"
    N5_ENV=""; fl=r3
    [ "$k" != base ] && { N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='${FS[$k]}'"; fl=r3$k; }
    TL_I $fl "n5_r3.py --plan plans3/flags.json --tag r${r}_$k --check --ck --rep 30" > tmp.out 2>> err.log; echo "rc=$?"
    grep '^{' tmp.out | sed "s/\"impl\"/\"flagset\": \"$k\", \"impl\"/" >> run.jsonl
  done
done
rm -f tmp.out; date; echo DONE
