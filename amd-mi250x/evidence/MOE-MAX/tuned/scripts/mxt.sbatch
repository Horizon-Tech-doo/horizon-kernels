#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:40:00
#SBATCH -J MOETRACE
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/MOE-MAX/raw/MXT-%j.out
# MOE-MAX tuned-baseline race. Arg 1: fwd|rev (arm order and case order). Source dir srct (landed moe_fused.py 8faee1ee).
# Routing draws data_t_7.pt, data_t_8.pt (ms_mk.py from data_w.pt, seeds 7 and 8). Tuned vLLM config: tuned/<vLLM file name>.json.
E=/scratch/project_XXXXXXXXX/tilelang/exp/MOE-MAX; cd $E; mkdir -p raw
J=$SLURM_JOB_ID; D=$E/srct; C=$E/cache_r_$J; mkdir -p $C
TJ=$(ls $E/tuned/E=256,N=512,device_name=*.json | head -1)
sg(){ $D/sgt.sh $D $C "$@"; }
REV=""; [ "$1" = rev ] && REV=--reverse
TS=${TS:-"1,2,3,4,8,16,24,1024,2048,2700,4096"}
# Arg 2 = z: add the arm vllmz (vLLM with the config tuned on zipf2 routing, tuned_zipf2/<vLLM file name>.json)
ARMS=g:ship,g:vllm,g:vllmt,g:same,g:vllm2; T2=""
if [ "$2" = z ]; then TZ=$(ls $E/tuned_zipf2/E=256,N=512,device_name=*.json | head -1); ARMS=g:ship,g:vllm,g:vllmt,g:vllmz,g:same,g:vllm2; md5sum "$TZ"; fi
echo "## start race $1 $(date +%T)"
md5sum $D/moe_fused.py $D/mxt.py "$TJ" $E/data_t_7.pt $E/data_t_8.pt
echo "## cfgcheck $(date +%T)"
SGT_ENV="VLLM_TUNED_CONFIG_FOLDER=$E/tuned" sg cfgcheck.py "$TJ" $TS | grep cfgcheck | tee -a raw/mxt_$J.jsonl
cs(){ python3 -c "import sys; cs=sys.argv[1].split(','); print(','.join(cs[::-1] if '$REV' else cs))" $1; }
rc(){ python3 -c "print(','.join([t+':'+r for t in '$1'.split(',') for r in ('uniform','zipf1','zipf2')]))"; }
echo "## prebuild $(date +%T)"
n=0; for t in ${TS//,/ }; do sg mxt.py --data $E/data_t_7.pt --cases $t:uniform --arms ship,same --build-only | grep 'kind": "prebuilt\|arm_failed\|rror' >> raw/mxt_$J.jsonl & n=$((n+1)); [ $((n % 4)) = 0 ] && wait; done; wait
for SEED in 7 8; do
  echo "## race seed $SEED $(date +%T)"
  sg mxt.py --data $E/data_t_$SEED.pt --tag s$SEED --tuned "$TJ" ${TZ:+--tuned2 "$TZ"} --cases $(cs $(rc $TS)) --arms $ARMS --rounds 6 --rep 60 $REV | tee -a raw/mxt_$J.jsonl | grep -c '"time"'
done
grep -c 'arm_failed' raw/mxt_$J.jsonl
rm -rf $C
echo "## end $(date +%T)"
