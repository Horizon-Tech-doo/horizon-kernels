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
# Landed file (Square gate/up on the clang >= 20 large tile): real repeat, scan across the boundary and above 8192, self-test.
DATA=$E/data_scan3.pt
TS="1025 1280 1536 2048 3072 4096 6144 7168 8192 12288 16384"
echo "## prebuild $(date +%T)"
for t in $TS; do vl --set r4 --cases $t:uniform --arms ship --no-vllm --build-only | grep '"kind"' >> raw/run_$J.jsonl & done; wait
echo "## prebuild done $(date +%T)"
R="--set r4 --arms ship --rounds 2 --rep 40"
for L in 19 39; do DATA=$E/real/s$L.pt; CR=8192:prefill,7168:prefill,6144:prefill
  echo "## L$L $(date +%T)"; vl $R --cases $CR | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
  echo "## L$L rev $(date +%T)"; vl $R --cases $CR --reverse | sed "s/^{/{\"layer\": $L, /" | tee -a raw/run_$J.jsonl
done
DATA=$E/real/data_real_L0.pt
echo "## L0 $(date +%T)"; vl $R --cases 8192:prefill | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
echo "## L0 rev $(date +%T)"; vl $R --cases 8192:prefill --reverse | sed "s/^{/{\"layer\": 0, /" | tee -a raw/run_$J.jsonl
DATA=$E/data_scan3.pt
CASES=$(python3 -c "print(','.join(f'{t}:{r}' for t in ($(echo $TS | tr ' ' ',')) for r in ('uniform','zipf1','zipf2')))")
A="--set r4 --arms ship --rounds 2 --rep 25 --cases $CASES"
echo "## scan $(date +%T)"; vl $A | tee -a raw/run_$J.jsonl
echo "## scan rev $(date +%T)"; vl $A --reverse | tee -a raw/run_$J.jsonl
ST=1,16,17,64,513,1024,1025,1536,2048,4096,8192,12288
echo "## self-test image $(date +%T)"; singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $D; python3 moe_fused.py --tokens $ST" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" | tee -a raw/selftest_image_$J.jsonl
echo "## self-test lumi $(date +%T)"; $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $D; python moe_fused.py --tokens 16,1025,8192" 2>&1 | grep -v "INFO\|Warning\|in memory cache" | tee -a raw/selftest_lumi_$J.jsonl
date
