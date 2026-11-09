#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=2
#SBATCH -c 14
#SBATCH --mem=120G
#SBATCH -t 00:20:00
#SBATCH -J N1-cap
# Real routing capture for the round 6 repeat: same script and sample as 23508902 and 23511393, on this track's own checkpoint copy. Then slices. Output to real6, so the kept route_stats in real/ stays.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p real6 raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; cd $E; python3 -I $D/n1_real.py $E/ckpt36 /scratch/project_XXXXXXXXX/data/prompts/pilot-val.jsonl $E/real6 1024 bfloat16" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" | tee -a raw/run_$SLURM_JOB_ID.jsonl
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "cd $E; for L in 19 39; do python3 -I $D/n1_mk5.py real6/data_real_L\$L.pt real6/s\$L.pt; done"
ls -la real6; date
