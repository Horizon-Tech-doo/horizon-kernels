#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:17:00
#SBATCH --job-name=C2P_c3
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# sbatch c3job.sh <arm regex> <rounds> [inputs]. Process order rotates per round: forms on fork-main, then hand-edited kernels on the stock wheel, each with its own CK and SDPA.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
RE=$1; R=$2; INP=${3:-randn0}; ST=${4:-1}
git -C $F rev-parse HEAD > fork_head.txt; git -C $F diff --stat > fork_diffstat.txt
date
for r in $(seq $ST $((ST + R - 1))); do
  for i in 0 1; do
    if [ $(( (i + r) % 2 )) = 0 ]; then
      echo "=== r$r fork $(date +%T)"; timeout 500 bash -c "source $S/cenv.sh; FM_I \"src/c3.py --tag r$r --re '$RE' --inputs $INP\"" > tmp.out 2>> err.log; echo rc=$?
    else
      echo "=== r$r land $(date +%T)"; timeout 200 bash -c "source $S/cenv.sh; WH_I \"src/c3.py --land --tag r$r --re '$RE' --inputs $INP\"" > tmp.out 2>> err.log; echo rc=$?
    fi
    grep '^{' tmp.out >> run.jsonl
  done
done
rm -f tmp.out; date; echo DONE
