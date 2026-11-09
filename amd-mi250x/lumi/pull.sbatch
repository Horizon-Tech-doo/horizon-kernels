#!/bin/bash
#SBATCH --job-name=vllm-pull
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Pull a vLLM ROCm image as a SIF. Build temp on node-local /tmp (RAM-backed tmpfs, hence --mem=120G);
# Lustre extraction + mksquashfs was too slow (two 1 h timeouts). Runs on a GPU node (GPUs unused) so it
# is billed in GPU-h (2/h) instead of the scarce CPU core-h. Output on /flash.
set -euo pipefail
TAG=${TAG:-v0.18.1}
PROJ=${PROJ:?set PROJ to your LUMI project}; OUT=${VLLM_IMAGES:-/flash/$PROJ/vllm-check/images}
T=/tmp/sing-$SLURM_JOB_ID
export SINGULARITY_TMPDIR=$T/tmp SINGULARITY_CACHEDIR=$T/cache APPTAINER_TMPDIR=$T/tmp APPTAINER_CACHEDIR=$T/cache
mkdir -p $SINGULARITY_TMPDIR $SINGULARITY_CACHEDIR $OUT
trap 'rm -rf $T' EXIT
df -h /tmp | tail -1
cd $OUT
time singularity pull --disable-cache --force vllm-rocm-$TAG.sif.partial docker://vllm/vllm-openai-rocm:$TAG
mv vllm-rocm-$TAG.sif.partial vllm-rocm-$TAG.sif
ls -la $OUT
