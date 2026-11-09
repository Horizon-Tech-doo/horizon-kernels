#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:09:00
#SBATCH --job-name=C2P_md
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# sbatch md.sh "<process order>" : MoE and decode on the image, one process per tree, 2 cycles, order given then reversed. wheel = stock 0.1.15, fork = main + series with T.exp as written, forkx = main + series with T.exp = T.__exp.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
git -C $F rev-parse HEAD > fork_head.txt; git -C $F diff --stat > fork_diffstat.txt
ORD=${1:-wheel fork forkx}; REV=$(echo $ORD | awk '{for(i=NF;i>0;i--)printf "%s ",$i}')
run(){ local t=$1 tag=$2; echo "=== $tag $t $(date +%T)"; case $t in
  wheel) timeout 150 bash -c "export FM_ENV=C2P_ARM=wheel; source $S/cenv.sh; WH_I \"src/md.py --tag $tag --seed $SLURM_JOB_ID\"" > tmp.out 2>> err.log;;
  fork) timeout 150 bash -c "export FM_ENV=C2P_ARM=fork; source $S/cenv.sh; FM_I \"src/md.py --tag $tag --seed $SLURM_JOB_ID\"" > tmp.out 2>> err.log;;
  forkx) timeout 150 bash -c "export FM_ENV='C2P_ARM=forkx C2P_FASTEXP=1'; source $S/cenv.sh; FM_I \"src/md.py --tag $tag --seed $SLURM_JOB_ID\"" > tmp.out 2>> err.log;;
  forkp3) timeout 150 bash -c "export FM_ENV='C2P_ARM=forkp3 C2P_M=m2 TILELANG_CACHE_DIR=$E/cache/IQ'; source $S/cenv.sh; FM_I \"src/md.py --tag $tag --seed $SLURM_JOB_ID --only decode\"" > tmp.out 2>> err.log;;
  forkxP) timeout 150 bash -c "export FM_ENV='C2P_ARM=forkxP C2P_FASTEXP=1 C2P_POST=P TILELANG_CACHE_DIR=$E/cache/IP'; source $S/cenv.sh; FM_I \"src/md.py --tag $tag --seed $SLURM_JOB_ID\"" > tmp.out 2>> err.log;;
  forkxO) timeout 150 bash -c "export FM_ENV='C2P_ARM=forkxO C2P_FASTEXP=1 C2P_POST=O TILELANG_CACHE_DIR=$E/cache/IO'; source $S/cenv.sh; FM_I \"src/md.py --tag $tag --seed $SLURM_JOB_ID\"" > tmp.out 2>> err.log;;
esac; echo rc=$?; grep '^{' tmp.out >> run.jsonl; }
for t in $ORD; do run $t c1; done
[ -n "${ONE_CYCLE:-}" ] || for t in $REV; do run $t c2; done
rm -f tmp.out; date; echo DONE
