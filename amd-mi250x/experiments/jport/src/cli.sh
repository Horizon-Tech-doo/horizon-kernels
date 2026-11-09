#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/JPORT/out/slurm-%j.out
# Landed module's own CLI, auto tile: gqa_fa.py L auto 128. vs_sdpa = same-process SDPA ms / ours ms, higher is better.
set -uo pipefail
ulimit -c 0
source /scratch/project_XXXXXXXXX/tilelang/exp/JPORT/src/env.sh
md5sum $TL_ROOT/scripts/gqa_fa.py
for L in 2560 3072 3584; do WH $X/cache/W "$TL_ROOT/scripts" "$TL_ROOT/scripts/gqa_fa.py $L auto 128" 2>/dev/null | grep '^{'; done
echo DONE
