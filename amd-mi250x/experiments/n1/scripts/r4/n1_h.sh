#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J N1-h
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v8h; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; CO=$E/co_$J; mkdir -p $CO
DATA=${DATA:-$E/data.pt}
cob(){ $TL "export TILELANG_CACHE_DIR=/tmp/c18_${J}_$1_$2 TRITON_CACHE_DIR=/tmp/c18_${J}_$1_$2/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c18_${J}_$1_$2" 2>&1 | grep "\"kind\"" | tee -a raw/run_$J.jsonl; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# T above 4096: the landed large tile reads over 1.0 at T = 8192 on real weights (23508944). Tile sweep, image then our stack, vLLM in the image.
ARMSV="ship,L_none,L_ilp,B1_ilp,B2_ilp,B3_ilp,B4_ilp,B5_ilp,B6_ilp,B7_ilp,C1_ilp,C2_ilp,C3_ilp,C4_ilp,C5_ilp,C6_ilp"
ARMSL="ship,L_ilp,L_none,B1_u20k,B3_u20k,B4_u20k,C1_u20k,C2_u20k,C3_u20k,C4_u20k,C5_u20k"
run(){ DATA=$1; echo "## $1 image $(date +%T)"; vl --set r4 --cases $2 --arms $ARMSV --rounds 2 --rep 25 | sed "s/^{/{\"data\": \"$3\", /" | tee -a raw/run_$J.jsonl; echo "## $1 image rev $(date +%T)"; vl --set r4 --cases $2 --arms $ARMSV --rounds 2 --rep 25 --reverse --no-vllm | sed "s/^{/{\"data\": \"$3\", /" | tee -a raw/run_$J.jsonl; echo "## $1 lumi $(date +%T)"; lu --set r4 --cases $2 --arms $ARMSL --rounds 2 --rep 25 | sed "s/^{/{\"data\": \"$3\", /" | tee -a raw/run_$J.jsonl; }
run $E/real/data_real_L19.pt 8192:prefill L19
run $E/data_scan.pt 6144:uniform,8192:uniform,8192:zipf1 scan
date
