#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p dev-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:08:00
#SBATCH -J MOESWEEP
#SBATCH -o /scratch/project_XXXXXXXXX/tilelang/exp/MOESWEEP/raw/MS-%j.out
# Time limit set per job at submit with -t: 10 min with builds, 6 cached.
# MOESWEEP. Args: mode (smoke|sweep|k1|err) rev|fwd [seed]. Compiled kernels shared by the track's jobs, chained so they never overlap.
E=/scratch/project_XXXXXXXXX/tilelang/exp/MOESWEEP; D=$E/src; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
J=$SLURM_JOB_ID; C=$E/cache; mkdir -p $C
sg(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; sys.argv[0]=\"$D/$1\"; runpy.run_path(\"$D/$1\", run_name=\"__main__\")' ${*:2}" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
MODE=$1; REV=""; [ "$2" = rev ] && REV=--reverse; ERV=""; [ "$2" = rev ] && ERV=--rev; SEED=${3:-7}
TSW="2600 2625 2650 2675 2700 2725 2750 2775 2800 2825 2850 2875 2900 2925 2950 2975 3000"
TALL="2450,2500,2550,$(echo $TSW | tr ' ' ,),7168,8193"
if [ ! -f $E/data_w.pt ]; then sg n1_mk.py $E/data_w.pt; fi
DATA=$E/data_s$SEED.pt
if [ ! -f $DATA ]; then sg ms_mk.py $E/data_w.pt $DATA $SEED $TALL; fi
echo "## start $MODE $2 $SEED $(date +%T)"
python3 -c "import hashlib,sys; print('landed md5', hashlib.md5(open('$D/moe_fused.py','rb').read()).hexdigest())"
k1p(){ sg msrun.py --data $DATA --stack vllm --set k1 "$@"; }
ALLK="ship,ship_z0,x_once,x_copy,xz_row,xz_dirty,xz_buf,pad2,gridt,g_once,g_dirty"
PK="ship,x_once,xz_row,xz_buf,pad2,gridt,g_once"   # sweep peak: does any isolation arm take T = 2700 under 1
case $MODE in
smoke)
  sg msrun.py --data $DATA --stack vllm --set r4 --arms ship --rounds 1 --rep 5 --percall --cases 2800:uniform | tee -a raw/run_$J.jsonl | grep -c '"time"'
  k1p --arms $ALLK --rounds 1 --rep 5 --cases 7168:uniform | tee -a raw/run_$J.jsonl | grep -c '"time"'
  sg ms_err.py --data $DATA --seed 21 --gs 1,2,4,8 --vllm | tee -a raw/err_$J.jsonl | grep -c err_all_T
  ;;
sweep)
  echo "## prebuild $(date +%T)"
  set -- $TSW; while [ $# -gt 0 ]; do g="$1:uniform"; shift; if [ $# -gt 0 ]; then g="$g,$1:uniform"; shift; fi; sg msrun.py --data $DATA --stack vllm --set r4 --cases $g --arms ship --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done; wait
  echo "## sweep $(date +%T)"
  CS=$(python3 -c "ts='$TSW'.split(); ts=ts[::-1] if '$REV' else ts; print(','.join(t+':uniform' for t in ts))")
  sg msrun.py --data $DATA --stack vllm --set r4 --arms ship --rounds 6 --rep 50 --percall --cases $CS $REV | tee -a raw/run_$J.jsonl | grep -c '"time"'
  ;;
k1)
  echo "## prebuild $(date +%T)"
  for t in 7168 8193; do k1p --cases $t:uniform --arms $ALLK --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done
  k1p --cases 2800:uniform --arms ship,gridt --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl &
  k1p --cases 2700:uniform --arms $PK --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & wait
  echo "## k1 $(date +%T)"
  k1p --arms $ALLK --rounds 4 --rep 40 --cases 7168:uniform,8193:uniform,7168:zipf1 $REV | tee -a raw/run_$J.jsonl | grep -c '"time"'
  k1p --arms ship,gridt --rounds 4 --rep 40 --percall --cases 2800:uniform $REV | tee -a raw/run_$J.jsonl | grep -c '"time"'
  k1p --arms $PK --rounds 4 --rep 40 --percall --cases 2700:uniform $REV | tee -a raw/run_$J.jsonl | grep -c '"time"'
  ;;
err)
  echo "## prebuild $(date +%T)"
  for G in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096; do sg ms_err.py --data $DATA --gs $G --build-only | grep -c built & done; wait
  echo "## err $(date +%T)"
  sg ms_err.py --data $DATA --seed $([ "$2" = rev ] && echo 22 || echo 21) --vllm $ERV | tee -a raw/err_$J.jsonl | grep -c err_all_T
  ;;
esac
echo "## end $(date +%T)"; date
