#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --job-name=C2P_s1
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# Tile sweep of the TileLang-only forms, image stack. One process per case group. Ratio ours/CK in the same process. No correctness repeats in the sweep; finalists are checked after.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
export C3_TILES=$S/sweep1.json C3_FORMS=SDRF,JDRF
date
for g in "d256_8192" "mha_4096" "d128_4096" "d128_8192" "d128_16384"; do
  echo "=== $g $(date +%T)"; timeout 500 bash -c "source $S/cenv.sh; FM_I \"src/c3.py --tag s1 --no-check --re '$g' --rep 20\"" > tmp.out 2>> err.log; echo rc=$?; grep '^{' tmp.out >> run.jsonl
done
rm -f tmp.out; date; echo DONE
