#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:12:00
#SBATCH -J N1-b128
# Experiment 2: T = 4096, bM 128 with the good down tile, our stack. Stacks alternate and the cycle order flips. Then the
# landed file's own self-test, then a clang 18 dump of the small tiles for reference (no timing).
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v6; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-b-$J; CV=$E/cache-bv-$J
lumi(){ $TL "export TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton; cd $E; python -I $D/n1run.py --data $E/data.pt --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$CV TRITON_CACHE_DIR=$CV/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $E/data.pt --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
A="--set j --arms $D/j7.json --tokens 4096 --routes uniform,zipf1,zipf2 --rounds 3 --rep 50"
echo "## cycle 1 lumi $(date +%T)"; lumi $A | tee -a raw/run_$J.jsonl
echo "## cycle 1 vllm $(date +%T)"; vl $A --no-tl | tee -a raw/run_$J.jsonl
echo "## cycle 2 vllm $(date +%T)"; vl $A --no-tl --reverse | tee -a raw/run_$J.jsonl
echo "## cycle 2 lumi $(date +%T)"; lumi $A --reverse | tee -a raw/run_$J.jsonl
echo "## cycle 3 lumi $(date +%T)"; lumi $A | tee -a raw/run_$J.jsonl
echo "## cycle 3 vllm $(date +%T)"; vl $A --no-tl | tee -a raw/run_$J.jsonl
echo "## self-test lumi $(date +%T)"; $TL "export TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton; cd $D; python moe_fused.py" 2>&1 | grep -v "INFO\|Warning\|in memory cache" | tee -a raw/selftest_$J.jsonl
echo "## clang 18 dump $(date +%T)"; lumi --set s --cases 16:uniform --rounds 0 --dump $E/asm_c18_$J | tee -a raw/run_$J.jsonl
python3 $D/n1asm.py $E/asm_c18_$J/*.dis > raw/asm_c18_$J.jsonl
date
