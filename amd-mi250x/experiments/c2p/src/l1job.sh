#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=C2P_l1
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# Finalist tiles on the LUMI stack (ROCm 6.2.4, clang 18). No CK in this stack; ratio is ours/SDPA in the same process. Forms on fork-main, 2 rounds.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
export C3_TILES=$S/confl.json C3_FORMS=SDRF,JDRF C3_STACK=lumi
for r in 1 2; do echo "=== r$r $(date +%T)"; timeout 250 bash -c "source $S/cenv.sh; FM_L \"src/c3.py --tag l$r\"" > tmp.out 2>> err.log; echo rc=$?; grep '^{' tmp.out >> run.jsonl; done
rm -f tmp.out; date; echo DONE
