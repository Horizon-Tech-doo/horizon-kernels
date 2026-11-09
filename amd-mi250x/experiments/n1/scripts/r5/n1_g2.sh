#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:35:00
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v10; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; mkdir -p $C $CL
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# T grid prototype (n1grid.py): power of two grid against per-T kernels, at grid-boundary T, plus exact-fit T. Landed tables on both stacks.
DATA=$E/data_grid.pt
TS="17 32 33 65 129 257 513 1024 1025 2049 4097"
echo "## prebuild $(date +%T)"
for t in $TS; do vl --set g --cases $t:uniform --arms ship,grid --no-vllm --build-only | grep '"kind"\|Error\|error' >> raw/run_$J.jsonl & done; wait
echo "## prebuild image done $(date +%T)"
for t in $TS; do lu --set g --cases $t:uniform --arms ship,grid --build-only | grep '"kind"\|Error\|error' >> raw/run_$J.jsonl & done; wait
echo "## prebuild done $(date +%T)"
CASES=$(python3 -c "print(','.join(f'{t}:{r}' for t in ($(echo $TS | tr ' ' ',')) for r in ('uniform','zipf1')))")
A="--set g --arms ship,grid --rounds 2 --rep 25 --cases $CASES"
echo "## image $(date +%T)"; vl $A | tee -a raw/run_$J.jsonl
echo "## image rev $(date +%T)"; vl $A --reverse | tee -a raw/run_$J.jsonl
echo "## lumi $(date +%T)"; lu $A | tee -a raw/run_$J.jsonl
echo "## lumi rev $(date +%T)"; lu $A --reverse | tee -a raw/run_$J.jsonl
for L in 19; do DATA=$E/real/g$L.pt; CR=$(python3 -c "print(','.join(f'{t}:prefill' for t in (17,33,129,513,1025,2049,4097)))")
  echo "## real L$L $(date +%T)"; vl --set g --arms ship,grid --rounds 2 --rep 25 --cases $CR | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## real L$L rev $(date +%T)"; vl --set g --arms ship,grid --rounds 2 --rep 25 --cases $CR --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
done
date
