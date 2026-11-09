#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:06:00
#SBATCH --job-name=n5co
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out4/slurm-%j.out
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets4.sh
O=$E/out4/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py $S/n5_r4.py $S/n5lib_r4.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
# D MHA 4k P160: lb 0/4 inside each process, flag sets between processes, image stack, clang 22.
n=0
for k in base ilp ilp base; do
  r=$((++n))
  echo "=== p$r $k $(date +%T)"
  N5_ENV=""; fl=r4
  [ "$k" != base ] && { N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='${FS[$k]}'"; fl=r4$k; }
  TL_I $fl "n5_r4.py --plan plans4/comb.json --tag p${r}_$k --check --ck --inputs randn0,ar,self,big" 2>> err.log | grep '^{' | sed "s/\"info\"/\"flagset\": \"$k\", \"info\"/; s/\"case\"/\"flagset\": \"$k\", \"case\"/" >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
done
date +%T; echo DONE
