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
CASES=8192:uniform,8192:zipf1,8192:zipf2,6144:uniform,6144:zipf1,4096:uniform,4096:zipf1,4096:zipf2,2048:uniform,2048:zipf1,2048:zipf2,1024:uniform,1024:zipf1,1024:zipf2
ARMSV=ship,LS_ilp,LS_none,LKS_ilp,LKS_none,MS_ilp@1024-1024,MS_none@1024-1024
ARMSL=ship,LS_u20k,LS_none,LKS_u20k,LKS_none,MS_u20k@1024-1024,MS_none@1024-1024
echo "## prebuild $(date +%T)"
for t in 8192 6144 4096 2048 1024; do vl --set r4 --cases $t:uniform --arms $ARMSV --no-vllm --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild image done $(date +%T)"
for t in 8192 4096 2048 1024; do lu --set r4 --cases $t:uniform --arms $ARMSL --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild lumi done $(date +%T)"
A="--set r4 --cases $CASES --rounds 2 --rep 25"
echo "## image rev $(date +%T)"; vl $A --arms $ARMSV --reverse | tee -a raw/run_$J.jsonl
echo "## image $(date +%T)"; vl $A --arms $ARMSV | tee -a raw/run_$J.jsonl
LC=8192:uniform,8192:zipf1,4096:uniform,4096:zipf1,2048:uniform,2048:zipf1,1024:uniform,1024:zipf1
echo "## lumi $(date +%T)"; lu --set r4 --cases $LC --rounds 2 --rep 25 --arms $ARMSL | tee -a raw/run_$J.jsonl
echo "## lumi rev $(date +%T)"; lu --set r4 --cases $LC --rounds 2 --rep 25 --arms $ARMSL --reverse | tee -a raw/run_$J.jsonl
date
