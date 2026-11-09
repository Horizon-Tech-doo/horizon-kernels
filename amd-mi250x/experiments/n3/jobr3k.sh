#!/bin/bash
#SBATCH --job-name=n3-r3k
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; mkdir -p realkv; RUNS=${1:-"dummy real"}
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
run(){ # tag layoutenv mode backend
 echo "#### $1 layout=$2 mode=$3 backend=$4 $(date +%T)"
 timeout ${5:-300} singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 TRITON_CACHE_DIR=$E/cache-v/triton HF_HOME=$E/hfhome ${2:+VLLM_KV_CACHE_LAYOUT=$2}; cd $E; python3 n3r3k.py $3 $E/realkv $4" 2>&1 | grep -E "KV cache layout|Using .* attention|backend|Backend|ROCM_ATTN|TRITON_ATTN|\"t\":|DONE|Error|error|Traceback|File \"|Unsupported" | grep -v "Warning\|amdsmi" | cut -c1-400 | sed "s/nid[0-9]*/NODE/g"
 echo "#### end $1 rc=$? $(date +%T)"
}
{ for r in $RUNS; do case $r in
 dummy) run def "" dummy "" 300; run tri "" dummy TRITON_ATTN 300; run hnd HND dummy "" 300; run tri_hnd HND dummy TRITON_ATTN 300;;
 real) run real "" real "" 1200;;
 realt) run real_def "" real "" 600; run real_tri "" real TRITON_ATTN 600;;
 esac; done; } > raw/r3k-$SLURM_JOB_ID.log 2>&1
echo "== end $(date +%T)"
