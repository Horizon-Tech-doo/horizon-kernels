#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -J N1-f
# Simulated FP8 and FP4 weights against landed fp16 in one process. No vendor FP8 or FP4 kernel exists in the harness: ratios are against our own fp16. Arg 1: rev.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
J=$SLURM_JOB_ID; C=$E/cache-$J; mkdir -p $C
f(){ DA=$1; LY=$2; CS=$3; shift 3; singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1fp.py\", run_name=\"__main__\")' --data $DA --layer $LY --cases $CS --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" | tee -a raw/fp_$J.jsonl ; }
REV=""; [ "$1" = rev ] && REV=--reverse
CS=$(python3 -c "print(','.join(f'{t}:{r}' for t in (1,16,64,256,1024,4096,8192) for r in ('uniform','zipf1')))")
echo "## syn $(date +%T)"; f $E/data_r6.pt syn $CS --rounds 4 --rep 25 $REV | grep -c '"time"'
for L in 19 39; do echo "## L$L $(date +%T)"; f $E/real6/s$L.pt $L 8192:prefill,4096:prefill,1024:prefill --rounds 4 --rep 25 $REV | grep -c '"time"'; done
rm -rf $C; date
