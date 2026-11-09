#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --job-name=C2P_c6
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# Six inputs on the image stack (tile bests only), then both tile bests on the LUMI stack without CK. Arm order rotates by round tag. Ratio: ours/CK in the same process, ours/SDPA.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
RE="(d256_8192_P32_b32|d128_4096|mha_4096_P160)"
INP=randn0,randn1,self,big,ar,ramp
date
for r in 1 2; do
  echo "=== r$r fork image $(date +%T)"; timeout 250 bash -c "source $S/cenv.sh; FM_I \"src/c3.py --tag i$r --re '$RE' --inputs $INP --rep 30\"" > tmp.out 2>> err.log; echo rc=$?; grep '^{' tmp.out >> run.jsonl
  echo "=== r$r land image $(date +%T)"; timeout 150 bash -c "source $S/cenv.sh; WH_I \"src/c3.py --land --tag i$r --re '$RE' --inputs $INP --rep 30\"" > tmp.out 2>> err.log; echo rc=$?; grep '^{' tmp.out >> run.jsonl
  echo "=== r$r fork lumi $(date +%T)"; timeout 120 bash -c "export C3_STACK=lumi; source $S/cenv.sh; FM_L \"src/c3.py --tag l$r --re '$RE'\"" > tmp.out 2>> err.log; echo rc=$?; grep '^{' tmp.out >> run.jsonl
  echo "=== r$r land lumi $(date +%T)"; timeout 120 bash -c "export C3_STACK=lumi; source $S/cenv.sh; WH_L \"src/c3.py --land --tag l$r --re '$RE'\"" > tmp.out 2>> err.log; echo rc=$?; grep '^{' tmp.out >> run.jsonl
done
rm -f tmp.out; date; echo DONE
