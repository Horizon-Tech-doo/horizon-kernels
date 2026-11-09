#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -J N1-s
# Settle resubmit: landed against FullRow and vLLM with per-call times on the 2049-4096 band and above, rep 40. Band also carries the mid tile (M_none), untimed above 2048 until now. Arg 1: rev.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v11; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
J=$SLURM_JOB_ID; C=$E/cache-$J; mkdir -p $C
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
DATA=$E/data_r6.pt; REV=""; [ "$1" = rev ] && REV=--reverse
echo "## prebuild $(date +%T)"
for g in 2049,2200,2400 2560,2800,3000 3072,3200,3400 3584,3800,4000 4095,4096; do CC=$(python3 -c "print(','.join(t+':uniform' for t in '$g'.split(',')))"); vl --set r4 --cases $CC --arms ship,row_large,M_none --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done
for g in 7168 12288 16384; do vl --set r4 --cases $g:uniform --arms ship,row_large --no-vllm --build-only | grep '"kind"\|rror' >> raw/run_$J.jsonl & done
wait
echo "## prebuild done $(date +%T)"
CB=$(python3 -c "print(','.join(f'{t}:uniform' for t in '2049 2200 2400 2560 2800 3000 3072 3200 3400 3584 3800 4000 4095 4096'.split())+',2400:zipf1,2800:zipf1,3072:zipf1,3072:zipf2,4096:zipf1')")
echo "## band $(date +%T)"; vl --set r4 --arms ship,row_large,M_none --rounds 4 --rep 40 --percall --cases $CB $REV | tee -a raw/run_$J.jsonl | grep -c '"time"'
echo "## hi $(date +%T)"; vl --set r4 --arms ship,row_large --rounds 4 --rep 40 --percall --cases 7168:uniform,7168:zipf1,12288:uniform,12288:zipf1,16384:uniform,16384:zipf1 $REV | tee -a raw/run_$J.jsonl | grep -c '"time"'
rm -rf $C; date
