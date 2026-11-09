#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:25:00
#SBATCH -J N1-o
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v7; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; CO=$E/co_$J; mkdir -p $CO
DATA=${DATA:-$E/data.pt}
cob(){ $TL "export TILELANG_CACHE_DIR=/tmp/c18_${J}_$1_$2 TRITON_CACHE_DIR=/tmp/c18_${J}_$1_$2/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c18_${J}_$1_$2" 2>&1 | grep "\"kind\"" | tee -a raw/run_$J.jsonl; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# Experiment 3, our stack, clang 18: ship (tile A + unroll 20000) against -misched=gcn-max-ilp and both. T=1 mid against ship. vLLM in the image, cycle order flipped.
CASES=1:uniform,1:zipf1,1:zipf2,16:uniform,16:zipf1,16:zipf2,64:uniform,64:zipf1,64:zipf2,256:uniform,256:zipf1,256:zipf2
ARMS="ship,A_ilp@16-256,A_ilpu@16-256,M_none@1-1"
A="--set r4 --arms $ARMS --cases $CASES --rounds 2 --rep 50"
echo "## 1 lumi $(date +%T)"; lu $A | tee -a raw/run_$J.jsonl
echo "## 1 vllm $(date +%T)"; vl $A --no-tl | tee -a raw/run_$J.jsonl
echo "## 2 vllm $(date +%T)"; vl $A --no-tl --reverse | tee -a raw/run_$J.jsonl
echo "## 2 lumi $(date +%T)"; lu $A --reverse | tee -a raw/run_$J.jsonl
date
