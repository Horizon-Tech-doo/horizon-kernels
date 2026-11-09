#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:35:00
#SBATCH -J N1-g
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v8; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; CO=$E/co_$J; mkdir -p $CO
DATA=${DATA:-$E/data.pt}
cob(){ $TL "export TILELANG_CACHE_DIR=/tmp/c18_${J}_$1_$2 TRITON_CACHE_DIR=/tmp/c18_${J}_$1_$2/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c18_${J}_$1_$2" 2>&1 | grep "\"kind\"" | tee -a raw/run_$J.jsonl; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# Real weights, real routing and real X of Qwen3.6-35B-A3B layers 0, 19, 39. Landed defaults on both stacks, clang 18 objects in the image, vLLM.
export E D J CO TL
b18(){ $TL "export TILELANG_CACHE_DIR=/tmp/p18_${J}_$1 TRITON_CACHE_DIR=/tmp/p18_${J}_$1/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"ship\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/p18_${J}_$1" 2>&1 | grep '"kind"'; }
export -f b18
echo 1,4,16,64,256,1024,4096,8192 | tr , '\n' | xargs -P 8 -I{} bash -c 'b18 {}' | tee -a raw/run_$J.jsonl
CASES=$(python3 -c "print(','.join([f'{t}:prefill' for t in (1,4,16,64,256,1024,4096,8192)]+[f'{t}:decode' for t in (1,4,16,64,256)]))")
A="--set r4 --codir $CO --rounds 2 --rep 40 --cases $CASES"
for L in 19 0 39; do
  DATA=$E/real/data_real_L$L.pt
  echo "## L$L 1 lumi $(date +%T)"; lu --arms ship $A | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L 1 vllm $(date +%T)"; vl --arms ship,co:ship $A | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L 2 vllm $(date +%T)"; vl --arms ship,co:ship $A --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L 2 lumi $(date +%T)"; lu --arms ship $A --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
done
date
