#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:06:00
#SBATCH --job-name=n5land5
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out5/slurm-%j.out
# Landed module through its real CLI: gqa_fa.py L auto D B G. Ratio vs_sdpa: same-process SDPA / ours in ms, higher is better. Fixed order. Script uses its own torch.randn inputs, seed 0.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src5/n5env5.sh
O=$E/out5/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
nodetag
go() { echo "=== $1 $2 $(date +%T)"; TL_$1 r5 "gqa_fa.py $2" 2>> err.log | grep '^{' | sed "s/^{/{\"s\": \"$1\", \"args\": \"$2\", /" >> run.jsonl; echo "rc=${PIPESTATUS[0]}"; }
for a in "2048 auto 128" "3072 auto 128" "4160 auto 128" "16384 auto 128" "2048 auto 128 1 1" "8192 auto 128 1 1"; do go I "$a"; done
for a in "2560 auto 128" "16384 auto 128" "2048 auto 128 1 1" "8192 auto 128 1 1"; do go L "$a"; done
date +%T; echo DONE
