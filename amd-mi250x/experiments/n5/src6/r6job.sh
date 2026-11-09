#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out6/slurm-%j.out
# $1: main | chk. Ratio: ours / same-process vendor, TF, higher is better. Shuffle seed has the job id. main: forms, hand-edited tiles, clang 18 lb.
set -uo pipefail
export FM_ENV=""
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src6/n5env6.sh
O=$E/out6/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py $S/n5_r5.py $S/n5lib_r5.py $S6/c6.py $C/tlfa.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
nodetag
J=$SLURM_JOB_ID
export C3_ARMS=$S6/plans/forms.json C3_SHUFFLE=$J C3_SDPA_END=1
fm() { # tag extra args
  local tag=$1; shift
  echo "=== forms $tag $(date +%T)"; FM6_I "c6.py --tag ${J}_$tag $*" 2>> err.log | grep '^{' >> forms.jsonl; echo "rc=${PIPESTATUS[0]}"
}
run() { # stack tag out args
  local st=$1 tag=$2 of=$3; shift 3
  echo "=== $st $tag $(date +%T)"; TL_$st r5 "n5_r5.py --tag ${J}_$tag $*" 2>> err.log | grep '^{' >> $of; echo "rc=${PIPESTATUS[0]}"
}
case "$1" in
main)
  for r in 1 2; do
    fm f$r
    run I t$r tile.jsonl --plan $S6/plans/tile22.json --check --ck
  done
  run L x1 l18.jsonl --plan $S6/plans/l18.json --check ;;
chk)   # $2 regex: winning arms on ar and big, with their b32 control; then rows 256 at the neighbouring table lengths
  fm c --re "'$2'" --inputs randn0,ar,big
  run I tb tileb.jsonl --plan $S6/plans/tile22b.json --check --ck ;;
esac
date +%T; echo DONE
