#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:40:00
#SBATCH -J MOETUNE
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/MOE-MAX/raw/MT-%j.out
# vLLM fused_moe Triton config tuning for the MOE-MAX shape. Arg 1: T list (comma separated). Arg 2: staged (s1 s2 fin, default)
# or full (full fin: every pruned config, as vLLM's tuner). Env: B1, B2, BF = s1 / s2 / full time budgets (s).
# Env ROUTE=zipf1|zipf2: tune on that skewed routing (default uniform, as vLLM's tuner); output files then carry the routing name.
# Per T: compile the stage's configs in 12 parallel processes (Triton cache), then time them in one process alone on the device.
E=/scratch/project_XXXXXXXXX/tilelang/exp/MOE-MAX; cd $E; mkdir -p raw
J=$SLURM_JOB_ID; D=$E/srct; C=$E/cache_t_$J; mkdir -p $C
RT=${ROUTE:-uniform}; RS=""; [ $RT != uniform ] && RS="_$RT"
sg(){ S_=$1; shift; $D/sgt.sh $D $C $S_ --route $RT "$@"; }
pb(){ # one prebuild shard, rerun past configs that abort the compiler
  for k in $(seq 1 60); do sg tune_moe.py --T $1 --stage $2 --out $3 --prebuild --shard $4/12 > $C/pb_$4.log 2>&1; grep -q '"kind": "prebuilt"' $C/pb_$4.log && break; done
  grep '"kind": "prebuilt"' $C/pb_$4.log | cut -c1-200; }
echo "## start tune $1 route $RT $(date +%T)"; md5sum $D/tune_moe.py
for T in ${1//,/ }; do
  O=$E/raw/tune_T${T}${RS}_$J.jsonl
  ST="s1 s2 fin"; [ "$2" = full ] && ST="full fin"
  for S in $ST; do
    if [ $S != fin ]; then
      echo "## T=$T $S prebuild $(date +%T)"
      for i in 0 1 2 3 4 5 6 7 8 9 10 11; do pb $T $S $O $i & done; wait
    fi
    echo "## T=$T $S time $(date +%T)"
    B=${B1:-420}; [ $S = s2 ] && B=${B2:-420}; [ $S = full ] && B=${BF:-900}
    for try in $(seq 1 40); do
      sg tune_moe.py --T $T --stage $S --out $O --budget $B | grep -v '"kind": "time"\|"kind": "fail"' | tail -4
      [ "$(grep -c "\"stage\": \"$S\".*stage_done\|stage_done.*\"stage\": \"$S\"" $O)" -ge 1 ] && break
      echo "## T=$T $S attempt $try ended without stage_done $(date +%T)"
    done
  done
  grep '"kind": "best"' $O | cut -c1-600; echo "## T=$T died-in-config: $(grep -c 'process died' $O)"
done
rm -rf $C
echo "## end $(date +%T)"
