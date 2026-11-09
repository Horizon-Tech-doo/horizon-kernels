#!/bin/bash
#SBATCH --job-name=n3-r5a
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N3/raw/%x-%j.out
E=/scratch/project_XXXXXXXXX/tilelang/exp/N3; cd $E; ROT=${1:-0}; CY=${2:-3}; SCR=${3:-n3r5a.py}; TAG=${4:-r5a}
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
image(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 $SCR image $*" 2>&1 | grep -v "Warning\|INFO\|WARNING\|amdsmi\|raw_cnt"; }
echo "== job $SLURM_JOB_ID rot $ROT parts $N3PARTS rev $N3REV $(date +%T)"
{ md5sum $E/gqa_decode.py /scratch/project_XXXXXXXXX/tilelang/scripts/gqa_decode.py $E/$SCR; echo "## job $SLURM_JOB_ID rot $ROT parts $N3PARTS rev $N3REV start $(date +%T)"
  for i in 0 1 2 3 4 5; do image 0 1 compile $i 6 & done; wait; echo "## compiled $(date +%T)"
  image $ROT $CY
  echo "## end $(date +%T)"
} 2>&1 | sed "s/nid[0-9]*/NODE/g" > raw/$TAG-$SLURM_JOB_ID.jsonl
echo "== end $(date +%T)"
