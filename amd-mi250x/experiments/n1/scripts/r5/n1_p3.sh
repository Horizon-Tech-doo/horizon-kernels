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
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
AV=ship,LS_ilp; AL=ship,LS_u20k
DATA=$E/real/s19.pt
echo "## prebuild $(date +%T)"
for t in 1024 1280 1536 2048 3072 4096 6144 7168 8192; do vl --set r4 --cases $t:prefill --arms $AV --no-vllm --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
for t in 1536 2048 4096 6144 7168 8192; do lu --set r4 --cases $t:prefill --arms $AL --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild done $(date +%T)"
CV=$(python3 -c "print(','.join(f'{t}:prefill' for t in (8192,7168,6144,4096,3072,2048,1536,1280,1024)))")
CL2=$(python3 -c "print(','.join(f'{t}:prefill' for t in (8192,7168,6144,4096,2048,1536)))")
A="--set r4 --rounds 2 --rep 40"
for L in 19 39; do DATA=$E/real/s$L.pt
  echo "## L$L vllm $(date +%T)"; vl $A --arms $AV --cases $CV | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L vllm rev $(date +%T)"; vl $A --arms $AV --cases $CV --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L lumi $(date +%T)"; lu $A --arms $AL --cases $CL2 | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L lumi rev $(date +%T)"; lu $A --arms $AL --cases $CL2 --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
done
DATA=$E/real/data_real_L0.pt
echo "## L0 vllm $(date +%T)"; vl $A --arms $AV --cases 8192:prefill | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
echo "## L0 vllm rev $(date +%T)"; vl $A --arms $AV --cases 8192:prefill --reverse | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
echo "## L0 lumi $(date +%T)"; lu $A --arms $AL --cases 8192:prefill | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
echo "## L0 lumi rev $(date +%T)"; lu $A --arms $AL --cases 8192:prefill --reverse | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
date
