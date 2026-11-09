#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --job-name=C2P_k3
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# K2b: image stack then LUMI stack, or reversed by $1 (I or L first).
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
git -C $F rev-parse HEAD > fork_head.txt
date
ST=${1:-I}
for s in $ST $([ $ST = I ] && echo L || echo I); do
  echo "=== $s $(date +%T)"; timeout 560 bash -c "source $S/cenv.sh; FM_$s \"src/k3.py --tag ${s}1\"" > $s.out 2> err_$s.log; echo rc=$?
done
date; echo DONE
