#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:06:00
#SBATCH --job-name=n5land4
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out4/slurm-%j.out
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets4.sh
O=$E/out4/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py $S/n5_r4.py $S/n5lib_r4.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
N5_ENV=""
TL_I land4 "land4.py" 2>> err.log | grep '^{' | sed 's/^{/{"s": "I", /' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
TL_L land4 "land4.py" 2>> err.log | grep '^{' | sed 's/^{/{"s": "L", /' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
TL_I land4 "gqa_fa.py 4096 - 128" 2>> err.log | grep '^{' > main_d128.jsonl; echo "rc=${PIPESTATUS[0]}"
date +%T; echo DONE
