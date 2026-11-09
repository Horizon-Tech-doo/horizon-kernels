#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=2
#SBATCH -c 14
#SBATCH --mem=120G
#SBATCH -t 00:40:00
#SBATCH -J N1-r
# Real routing capture: Qwen3.6-35B-A3B forward over pilot-val conversations in the image, two GCDs, then data_real_L*.pt.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v7; cd $E; mkdir -p real raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; cd $E; python3 -I $D/n1_real.py /scratch/project_XXXXXXXXX/models/Qwen3.6-35B-A3B /scratch/project_XXXXXXXXX/data/prompts/pilot-val.jsonl $E/real 1024 ${1:-bfloat16}" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" | tee -a raw/run_$SLURM_JOB_ID.jsonl
ls -la real; date
