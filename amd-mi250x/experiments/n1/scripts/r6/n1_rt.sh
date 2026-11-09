#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -J N1-rt
# Real data, landed import against vLLM in one process, with per-call times. L19, L39. Arg 1: rev reverses the first pass. Arg 2: extra arms.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
J=$SLURM_JOB_ID; C=$E/cache-$J; mkdir -p $C
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
A=ship${2:+,$2}; R1=""; R2=--reverse; [ "$1" = rev ] && { R1=--reverse; R2=""; }
DATA=$E/real6/s19.pt
echo "## prebuild $(date +%T)"
for g in 8192:prefill,1024:prefill 7168:prefill,1280:prefill 6144:prefill,1536:prefill 4096:prefill,2048:prefill 3072:prefill; do vl --set r4 --cases $g --arms $A --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done; wait
echo "## prebuild done $(date +%T)"
HI=8192:prefill,7168:prefill,6144:prefill; LO=4096:prefill,3072:prefill,2048:prefill,1536:prefill,1280:prefill
for L in 19 39; do DATA=$E/real6/s$L.pt
  echo "## L$L hi $(date +%T)"; vl --set r4 --arms $A --rounds 4 --rep 40 --percall --cases $HI $R1 | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl | grep -c '"time"'
  echo "## L$L hi second $(date +%T)"; vl --set r4 --arms $A --rounds 4 --rep 40 --percall --cases $HI $R2 | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl | grep -c '"time"'
  echo "## L$L lo $(date +%T)"; vl --set r4 --arms $A --rounds 4 --rep 40 --cases $LO $R1 | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl | grep -c '"time"'
done
DC=1:decode,4:decode,16:decode,64:decode,256:decode
for L in 19 39; do DATA=$E/real6/data_real_L$L.pt
  echo "## L$L decode $(date +%T)"; vl --set r4 --arms $A --rounds 4 --rep 40 --cases $DC $R1 | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl | grep -c '"time"'
done
rm -rf $C; date
