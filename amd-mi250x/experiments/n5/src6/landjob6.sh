#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:05:00
#SBATCH --job-name=N5-r6land
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out6/slurm-%j.out
# Landed module through its real CLI: gqa_fa.py L auto D. Clang 22, the changed d128_tile entries. Ratio vs_sdpa: same-process SDPA / ours in ms, higher is better. Fixed order. The script uses its own torch.randn inputs, seed 0. No CK in the script.
set -uo pipefail
export FM_ENV=""
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src6/n5env6.sh
S=$S6
O=$E/out6/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
nodetag
go() { echo "=== $1 $2 $(date +%T)"; TL_$1 r5 "gqa_fa.py $2" 2>> err.log | grep '^{' | sed "s/^{/{\"s\": \"$1\", \"args\": \"$2\", /" >> run.jsonl; echo "rc=${PIPESTATUS[0]}"; }
for L in 6144 8192 10240 12288 16384; do go I "$L auto 128"; done
date +%T; echo DONE
