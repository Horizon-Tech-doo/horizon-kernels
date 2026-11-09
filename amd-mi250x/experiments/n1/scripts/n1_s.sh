#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:15:00
#SBATCH -J N1-s
# Experiment 1, image only: disassembly of every arm first, then 2 cycles of 2 rounds, cycle 2 with the arm order reversed.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v6; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
C=$E/cache-s-$SLURM_JOB_ID; J=$SLURM_JOB_ID
[ -f data.pt ] || /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd $E; python -I n1_mk.py $E/data.pt" 2>&1 | grep -v Warning
ENVV="export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP"
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "$ENVV; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $E/data.pt --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
CASES=16:uniform,16:zipf1,16:zipf2,64:uniform,64:zipf1,256:uniform,256:zipf1
echo "## dump $(date +%T)"; vl --set s --cases $CASES --rounds 0 --no-vllm --dump $E/asm_s_$J | tee -a raw/run_$J.jsonl
python3 $D/n1asm.py $E/asm_s_$J/*.dis > raw/asm_s_$J.jsonl
echo "## cycle 1 $(date +%T)"; vl --set s --cases $CASES --rounds 2 --rep 50 | tee -a raw/run_$J.jsonl
echo "## cycle 2 reversed $(date +%T)"; vl --set s --cases $CASES --rounds 2 --rep 50 --reverse | tee -a raw/run_$J.jsonl
date
