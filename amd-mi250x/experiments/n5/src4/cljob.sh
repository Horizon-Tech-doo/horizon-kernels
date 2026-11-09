#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=n5cl
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out4/slurm-%j.out
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
source $S/flagsets4.sh
O=$E/out4/$SLURM_JOB_ID; mkdir -p $O; cd $O
ulimit -c 0
date +%T; md5sum $S/gqa_fa.py $S/n5_r4.py $S/n5lib_r4.py
rocm-smi --showuniqueid 2>/dev/null | grep -i "unique" | sed -E "s/^.*Unique ID: *//" | sed "s/^/gcd_unique_id /"
# C clang 18 flags, LUMI stack, processes rotate through flag sets. No CK on this stack.
ORDER=(base ilp clause pad)
for r in 1 2 3; do
  for i in 0 1 2 3; do
    k=${ORDER[$(( (i + r - 1) % 4 ))]}
    echo "=== r$r $k $(date +%T)"
    N5_ENV=""; fl=r4
    [ "$k" != base ] && { N5_ENV="HIPCC_COMPILE_FLAGS_APPEND='${FS[$k]}'"; fl=r4$k; }
    [ -d $E/cache/L${fl}_$PH ] || { echo "skip $k: no cache"; continue; }
    TL_L $fl "n5_r4.py --plan plans4/cl18.json --tag r${r}_$k --check" 2>> err.log | grep '^{' | sed "s/\"info\"/\"flagset\": \"$k\", \"info\"/; s/\"case\"/\"flagset\": \"$k\", \"case\"/" >> run.jsonl; echo "rc=${PIPESTATUS[0]}"
  done
done
date +%T; echo DONE
