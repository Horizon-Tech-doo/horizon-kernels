#!/bin/bash
#SBATCH --job-name=up-wvsplitk
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=40G
#SBATCH --time=00:15:00
# Runs the stand-alone reproduction in the stock vLLM 0.30.0 ROCm image on one MI250X GCD. Nothing of this project's code is on the path.
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif; D=/scratch/project_XXXXXXXXX/tilelang/exp/UPSTREAM
rocm-smi --showproductname 2>/dev/null | grep -i "card series\|card model" | head -2; uname -r; singularity exec $SIF bash -c "cat /opt/rocm/.info/version 2>/dev/null; pip list 2>/dev/null | grep -i -E '^(vllm|torch|triton|amd.aiter|aiter) '"
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1; python3 $D/repro_wvsplitk.py" 2>&1 | grep -v "^W1\|agent.cpp"
