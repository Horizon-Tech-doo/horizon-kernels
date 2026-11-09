#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:25:00
#SBATCH -J N1-b
# Decode-size weight bandwidth on the landed import with vLLM in the same process, then T-row sum grid. Arg 1: "rev".
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
J=$SLURM_JOB_ID; C=$E/cache-$J; mkdir -p $C
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
DATA=$E/data_r6.pt; REV=""; [ "$1" = rev ] && REV=--reverse
DEC="1 2 4 8 16 32 64 128 256"
GR="17 2049 4097 8193"
echo "## prebuild $(date +%T)"
for g in 1,2,4 8,16,32 64,128,256; do CC=$(python3 -c "print(','.join(t+':uniform' for t in '$g'.split(',')))"); vl --set r4 --cases $CC --arms ship --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done
for t in $GR; do vl --set g --cases $t:uniform --arms ship,grid,gridt,gride --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done
wait
echo "## prebuild done $(date +%T)"
CD=$(python3 -c "print(','.join(f'{t}:{r}' for t in '$DEC'.split() for r in ('uniform','zipf1','zipf2')))")
echo "## bw $(date +%T)"; vl --set r4 --arms ship --rounds 4 --rep 40 --cases $CD $REV | tee -a raw/run_$J.jsonl | grep -c time
CG=$(python3 -c "print(','.join(f'{t}:{r}' for t in '$GR'.split() for r in ('uniform','zipf1')))")
echo "## grid $(date +%T)"; vl --set g --arms ship,grid,gridt,gride --rounds 4 --rep 25 --cases $CG $REV | sed 's/^{/{"grid": 1, /' | tee -a raw/run_$J.jsonl | grep -c time
rm -rf $C; date
