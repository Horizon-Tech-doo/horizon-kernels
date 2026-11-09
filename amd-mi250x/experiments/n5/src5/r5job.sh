#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out5/slurm-%j.out
# $1: grid | val | attr | sweep | conf | gridchk | land. Ratio: ours / same-process vendor, TF, higher is better. Arm order shuffled, seed = tag/case/input with the job id in the tag.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src5/n5env5.sh
O=$E/out5/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py $S/n5_r5.py $S/n5lib_r5.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
nodetag
J=$SLURM_JOB_ID
run() { # stack flavour tag args
  local st=$1 fl=$2 tag=$3; shift 3
  echo "=== $st $tag $(date +%T)"; TL_$st $fl "n5_r5.py --tag ${J}_$tag $*" 2>> err.log | grep '^{' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
}
case "$1" in
grid)
  for r in 1 2; do for L in 2816 3072 3584; do run I r5 r${r}_g$L --plan plans5/grid$L.json --check --ck; done; done ;;
gridchk)
  run I r5 chk --plan plans5/$2 --check --ck --inputs randn0,ar,big ;;
val)
  for r in 1 2; do
    run I r5 r${r}_v22 --plan plans5/v22.json --check --ck
    run I r5 r${r}_ba --plan plans5/batch.json --check --ck --inputs ar,big
    run L r5 r${r}_v18 --plan plans5/v18.json --check
  done ;;
attr)
  for r in 1 2 3 4 5; do
    N5_ENV=""; run L r5 r${r}_base --plan plans5/attr.json --check
    N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='-mllvm -misched=gcn-max-ilp'"; run L r5ilp r${r}_ilp --plan plans5/attr_ilp.json --check
  done ;;
la18)
  for r in 1 2; do run L r5 r${r}_la --plan plans5/la18.json --check; done ;;
valb)
  for r in 1 2; do
    run I r5 r${r}_v22b --plan plans5/v22b.json --check --ck
    run L r5 r${r}_v18b --plan plans5/v18b.json --check
  done ;;
l18)
  run L r5 l18 --plan plans5/l18.json --check ;;
sweep)
  export C3_TILES=$S/plans5/sweep16.json C3_FORMS=SDRF,JDRF C3_SHUFFLE=$J FM_ENV=""
  echo "=== sweep $(date +%T)"; FM5_L "--tag s --no-check --re d128_16384 --rep 20" 2>> err.log | grep '^{' >> run.jsonl; echo "rc=${PIPESTATUS[0]}" ;;
conf)
  export C3_ARMS=$S/plans5/conf.json C3_SHUFFLE=$J C3_SDPA_END=1 FM_ENV=""
  for r in 1 2; do echo "=== conf r$r $(date +%T)"; FM5_L "--tag c$r --re d128_16384" 2>> err.log | grep '^{' >> run.jsonl; echo "rc=${PIPESTATUS[0]}"; done ;;
esac
date +%T; echo DONE
