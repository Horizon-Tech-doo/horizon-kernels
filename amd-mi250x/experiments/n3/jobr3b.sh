#!/bin/bash
#SBATCH --job-name=n3-r3b
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=00:08:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
( while true; do echo "$(date +%s.%N) $(/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh 'rocm-smi --showclocks' 2>/dev/null | grep -E 'sclk|mclk' | tr -s ' \t' ' ' | tr '\n' ';')"; sleep 1; done > raw/clocks-$SLURM_JOB_ID.log ) & W=$!
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 n3r3b.py" 2>&1 | grep -v "Warning\|INFO\|WARNING\|amdsmi\|raw_cnt" | sed "s/nid[0-9]*/NODE/g" > raw/r3b-$SLURM_JOB_ID.jsonl
kill $W; echo "== end $(date +%T)"
