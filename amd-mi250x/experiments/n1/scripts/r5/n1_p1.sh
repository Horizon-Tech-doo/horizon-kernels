#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J N1-p
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v9; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J
DATA=$E/data_scan.pt
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
CASES=${CASES:-8192:uniform,8192:zipf1,6144:uniform,6144:zipf1,3072:uniform}
TS=$(echo $CASES | tr ',' '\n' | cut -d: -f1 | sort -un)
ARMSV=${ARMSV:-ship,B6_ilp,B6S_ilp,B6S_none,B6SS_ilp,B6SS_none,LS_ilp,LS_none,LDS_ilp}
ARMSL=${ARMSL:-ship,B6_u20k,B6S_u20k,B6S_none,B6SS_u20k,B6SS_none,LS_u20k,LS_none,LDS_u20k}
echo "## prebuild $(date +%T)"
for t in $TS; do vl --set r4 --cases $t:uniform --arms $ARMSV --no-vllm --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild image done $(date +%T)"
for t in $TS; do lu --set r4 --cases $t:uniform --arms $ARMSL --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild lumi done $(date +%T)"
A="--set r4 --cases $CASES --rounds 2 --rep 25"
echo "## image $(date +%T)"; vl $A --arms $ARMSV | tee -a raw/run_$J.jsonl
echo "## image rev $(date +%T)"; vl $A --arms $ARMSV --reverse | tee -a raw/run_$J.jsonl
echo "## lumi $(date +%T)"; lu $A --arms $ARMSL | tee -a raw/run_$J.jsonl
echo "## lumi rev $(date +%T)"; lu $A --arms $ARMSL --reverse | tee -a raw/run_$J.jsonl
date
