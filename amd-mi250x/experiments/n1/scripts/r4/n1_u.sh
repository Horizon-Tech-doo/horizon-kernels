#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:30:00
#SBATCH -J N1-u
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v8h; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; CO=$E/co_$J; mkdir -p $CO
DATA=${DATA:-$E/data.pt}
cob(){ $TL "export TILELANG_CACHE_DIR=/tmp/c18_${J}_$1_$2 TRITON_CACHE_DIR=/tmp/c18_${J}_$1_$2/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c18_${J}_$1_$2" 2>&1 | grep "\"kind\"" | tee -a raw/run_$J.jsonl; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# Sweet spots of vLLM's bM 128 tile: N/256 just under 128 or 256, T near 3072 to 4000 and 7000 to 8192, and above 8192. Landed defaults, both stacks. Then real-routing slices at off-sample T.
$TL "cd $E; python -I $D/n1_mk3.py $E/data.pt $E/data_scan2.pt 2800,3072,3300,3500,3700,3800,3900,4000,6600,7000,7500,8000,12288,16384; python -I $D/n1_mk4.py $E/real/data_real_L19.pt $E/real/x19.pt; python -I $D/n1_mk4.py $E/real/data_real_L39.pt $E/real/x39.pt" 2>&1 | grep -v Warning
A="--set r4 --arms ship,B6_ilp@3000-4000 --rounds 2 --rep 30"
AL="--set r4 --arms ship --rounds 2 --rep 30"
C2=$(python3 -c "print(','.join(f'{t}:{r}' for t in (2800,3072,3300,3500,3700,3800,3900,4000,6600,7000,7500,8000,12288,16384) for r in ('uniform','zipf1')))")
CR=$(python3 -c "print(','.join(f'{t}:prefill' for t in (8,32,128,512,513,2048,3072,3584,6144,7168)))")
DATA=$E/data_scan2.pt
echo "## syn vllm $(date +%T)"; vl $A --cases $C2 | sed 's/^{/{"data": "syn", /' | tee -a raw/run_$J.jsonl
echo "## syn lumi $(date +%T)"; lu $AL --cases $C2 | sed 's/^{/{"data": "syn", /' | tee -a raw/run_$J.jsonl
for L in 19 39; do DATA=$E/real/x$L.pt
echo "## L$L vllm $(date +%T)"; vl $AL --cases $CR | sed "s/^{/{\"data\": \"L$L\", /" | tee -a raw/run_$J.jsonl
echo "## L$L lumi $(date +%T)"; lu $AL --cases $CR | sed "s/^{/{\"data\": \"L$L\", /" | tee -a raw/run_$J.jsonl
done
DATA=$E/data_scan2.pt
echo "## syn2 lumi $(date +%T)"; lu $AL --cases $C2 --reverse | sed 's/^{/{"data": "syn", /' | tee -a raw/run_$J.jsonl
echo "## syn2 vllm $(date +%T)"; vl $A --cases $C2 --reverse | sed 's/^{/{"data": "syn", /' | tee -a raw/run_$J.jsonl
date
