#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:20:00
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v10; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; mkdir -p $C $CL
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# Mid tile past T = 1024 on clang 22: it beats the large tile by 10% at T = 1024 uniform (23510941). Boundary check, same process as vLLM.
TS="1025 1280 1536 2048"
DATA=$E/data_scan3.pt
echo "## prebuild $(date +%T)"
for t in $TS; do vl --set r4 --cases $t:uniform --arms ship,M_none --no-vllm --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild done $(date +%T)"
CS=$(python3 -c "print(','.join(f'{t}:{r}' for t in (1025,1280,1536,2048) for r in ('uniform','zipf1','zipf2')))")
S="--set r4 --arms ship,M_none --rounds 2 --rep 40 --cases $CS"
echo "## syn $(date +%T)"; vl $S | tee -a raw/run_$J.jsonl
echo "## syn rev $(date +%T)"; vl $S --reverse | tee -a raw/run_$J.jsonl
for L in 19 39; do DATA=$E/real/s$L.pt; CR=2048:prefill,1536:prefill,1280:prefill
  echo "## L$L $(date +%T)"; vl --set r4 --arms ship,M_none --rounds 2 --rep 40 --cases $CR | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L rev $(date +%T)"; vl --set r4 --arms ship,M_none --rounds 2 --rep 40 --cases $CR --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
done
date
