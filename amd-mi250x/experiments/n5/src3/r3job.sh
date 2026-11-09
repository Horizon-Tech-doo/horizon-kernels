#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --job-name=n5r3
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out3/slurm-%j.out
# $1 plan, $2 stack I or L, $3 rounds, $4 extra args for n5_r3.py (quoted)
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out3/$SLURM_JOB_ID; mkdir -p $O; cd $O
N5_ENV=""
date; echo "plan=$1 stack=$2 rounds=$3 extra=$4"; md5sum $S/gqa_fa.py $S/n5_r3.py $S/plans3/$1.json
( while true; do echo "$(date +%T) $(rocm-smi --showpower --showclocks 2>/dev/null | grep -E "sclk|Power" | tr -s " " | cut -c1-60 | tr "\n" ";")"; sleep 15; done ) > clocks.log 2>&1 &
CLK=$!; python3 /scratch/project_XXXXXXXXX/tilelang/exp/N4/pollm.py metrics.jsonl 0.05 & POLL=$!
X=$!; trap "kill $CLK $POLL 2>/dev/null" EXIT
for r in $(seq 1 $3); do
  echo "=== r$r $(date +%T)"
  if [ "$2" = I ]; then TL_I r3 "n5_r3.py --plan plans3/$1.json --tag r$r --check $4" > tmp.out 2>> err.log; else TL_L r3 "n5_r3.py --plan plans3/$1.json --tag r$r --check $4" > tmp.out 2>> err.log; fi; echo "rc=$?"
  grep '^{' tmp.out >> run.jsonl
done
rm -f tmp.out; date; echo DONE
