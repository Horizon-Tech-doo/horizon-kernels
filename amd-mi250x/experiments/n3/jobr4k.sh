#!/bin/bash
#SBATCH --job-name=n3-r4k
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=00:06:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
{ echo "## job $SLURM_JOB_ID start $(date +%T)"
 timeout 320 singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TRITON_CACHE_DIR=$E/cache-v/triton HF_HOME=$E/hfhome; cd $E; python3 n3r4k.py $1" 2>&1 | grep -E "KV cache|KV cache layout|Using .* attention|Maximum concurrency|backend|Backend|ROCM_ATTN|TRITON_ATTN|\"t\":|DONE|Error|error|Traceback|File \"|Unsupported|tensor_parallel|world_size|rank" | grep -v "Warning\|amdsmi" | cut -c1-500 | sed -E "s/nid[0-9]*/NODE/g; s/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(:[0-9]+)?/ADDR/g"
 echo "## end rc=$? $(date +%T)"; } > raw/r4k-$SLURM_JOB_ID.log 2>&1
