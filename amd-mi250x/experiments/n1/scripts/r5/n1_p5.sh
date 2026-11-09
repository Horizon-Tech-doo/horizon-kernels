#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:30:00
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v10; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; mkdir -p $C $CL
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# Repeat of the Square headline on the landed file, FullRow control in the same process; job alone. Then a T grid repeat.
TS="3072 6144 7168 8192 12288 16384"
DATA=$E/data_scan3.pt
echo "## prebuild $(date +%T)"
for t in $TS; do vl --set r4 --cases $t:uniform --arms ship,row_large --no-vllm --build-only | grep '"kind"' >> raw/run_$J.jsonl & done
DATA=$E/data_grid2.pt
for t in 17 33 2049 4097 8193; do vl --set g --cases $t:uniform --arms ship,grid --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done; wait
echo "## prebuild done $(date +%T)"
R="--set r4 --arms ship,row_large --rounds 2 --rep 40"
for L in 19 39; do DATA=$E/real/s$L.pt; CR=8192:prefill,7168:prefill,6144:prefill
  echo "## L$L $(date +%T)"; vl $R --cases $CR | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L rev $(date +%T)"; vl $R --cases $CR --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
done
DATA=$E/real/data_real_L0.pt
echo "## L0 $(date +%T)"; vl $R --cases 8192:prefill | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
echo "## L0 rev $(date +%T)"; vl $R --cases 8192:prefill --reverse | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
DATA=$E/data_scan3.pt
CS=8192:uniform,8192:zipf1,7168:uniform,7168:zipf1,3072:uniform,12288:uniform,12288:zipf1,16384:uniform,16384:zipf1
S="--set r4 --arms ship,row_large --rounds 2 --rep 25 --cases $CS"
echo "## syn $(date +%T)"; vl $S | tee -a raw/run_$J.jsonl
echo "## syn rev $(date +%T)"; vl $S --reverse | tee -a raw/run_$J.jsonl
DATA=$E/data_grid2.pt
CG=$(python3 -c "print(','.join(f'{t}:{r}' for t in (17,33,2049,4097,8193) for r in ('uniform','zipf1')))")
G="--set g --arms ship,grid --rounds 2 --rep 25 --cases $CG"
echo "## grid $(date +%T)"; vl $G | sed 's/^{/{"grid": 1, /' | tee -a raw/run_$J.jsonl
echo "## grid rev $(date +%T)"; vl $G --reverse | sed 's/^{/{"grid": 1, /' | tee -a raw/run_$J.jsonl
date
