#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:25:00
#SBATCH -J N1-c
# Round 4 experiment 1, image process only: clang 18 code objects (whole MoE call through hipModule) against the mid tile, vLLM and -misched arms. Dump first, 2 cycles of 2 rounds, cycle 2 reversed.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v7; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-c-$J; CO=$E/co_c_$J; mkdir -p $CO
[ -f data.pt ] || $TL "cd $E; python -I $D/n1_mk.py $E/data.pt; python -I $D/n1_mk2.py $E/data.pt" 2>&1 | grep -v Warning
cob(){ $TL "export TILELANG_CACHE_DIR=/tmp/c18_${J}_$1_$2 TRITON_CACHE_DIR=/tmp/c18_${J}_$1_$2/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c18_${J}_$1_$2" 2>&1 | grep '"kind"' | tee -a raw/run_$J.jsonl; }
ENVV="export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP"
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "$ENVV; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $E/data.pt --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
vlco(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=/tmp/c22_${J}_$1 TRITON_CACHE_DIR=/tmp/c22_${J}_$1/t HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c22_${J}_$1" 2>&1 | grep '"kind"' | tee -a raw/run_$J.jsonl; }
echo "## code objects $(date +%T)"
for T in 16 64 256; do
  vlco $T mid; mv $CO/T${T}_mid $CO/T${T}_c22mid    # clang 22, loader control
  for tag in small+u20000 small mid; do cob $T $tag; done     # clang 18
done
CASES=16:uniform,16:zipf1,16:zipf2,64:uniform,64:zipf1,256:uniform,256:zipf1
ARMS="mid,co:c22mid,co:small+u20000,co:small,co:mid,A_none,A_ilp,A_mmc,A_ilpd,D_none,D_ilp"
A="--set r4 --arms $ARMS --codir $CO --cases $CASES"
echo "## dump $(date +%T)"; vl $A --rounds 0 --no-vllm --dump $E/asm_c_$J | tee -a raw/run_$J.jsonl
python3 $D/n1asm.py $E/asm_c_$J/*.dis > raw/asm_c_$J.jsonl
echo "## cycle 1 $(date +%T)"; vl $A --rounds 2 --rep 50 | tee -a raw/run_$J.jsonl
echo "## cycle 2 reversed $(date +%T)"; vl $A --rounds 2 --rep 50 --reverse | tee -a raw/run_$J.jsonl
date
