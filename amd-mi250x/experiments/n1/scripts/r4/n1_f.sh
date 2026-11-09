#!/bin/bash
#SBATCH -A project_XXXXXXXXX
#SBATCH -p small-g
#SBATCH --gpus=1
#SBATCH -c 14
#SBATCH --mem=60G
#SBATCH -t 00:50:00
#SBATCH -J N1-f
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; D=$E/v8; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VP=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages
TL=/scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh
J=$SLURM_JOB_ID; C=$E/cache-$J; CL=$E/cachel-$J; CO=$E/co_$J; mkdir -p $CO
DATA=${DATA:-$E/data.pt}
cob(){ $TL "export TILELANG_CACHE_DIR=/tmp/c18_${J}_$1_$2 TRITON_CACHE_DIR=/tmp/c18_${J}_$1_$2/t; cd $E; python -I -c 'import sys,runpy; sys.argv=[\"$D/n1co.py\",\"$CO\",\"$1\",\"$2\"]; sys.path.insert(0,\"$D\"); runpy.run_path(\"$D/n1co.py\",run_name=\"__main__\")'; rm -rf /tmp/c18_${J}_$1_$2" 2>&1 | grep "\"kind\"" | tee -a raw/run_$J.jsonl; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); import runpy; runpy.run_path(\"$D/n1run.py\", run_name=\"__main__\")' --data $DATA --stack vllm $*" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" ; }
lu(){ $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1run.py --data $DATA --stack lumi $*" 2>&1 | grep -v "INFO\|Warning\|in memory cache" ; }
# Landed file through the T scan. Image process: landed defaults (clang 22), the same table on clang 18 objects, vLLM. Our stack: landed defaults and tile A + unroll between 257 and 511. Both stacks' self-test on landed file, sampled and off-sample T.
DATA=$E/data_scan.pt; TLIST=2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,31,32,33,36,40,44,48,52,56,60,63,64,65,72,80,96,112,127,128,129,160,192,224,255,256,257,300,320,384,400,448,511,512,513,600,640,700,768,800,900,1000,1023,1024,1025,1200,1536,2047,2048,2049,3072,4095,4096,4097,6144,8192,1
[ -f $DATA ] || exit 3
echo "## prebuild $(date +%T)"; mkdir -p $C $CL
$TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $E; python -I $D/n1pre.py 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,31,32,33,36,40,44,48,52,56,60,63,64,65,72,80,96,112,127,128,129,160,192,224,255,256,257,300,320,384,400,448,511,512,513,600,640,700,768,800,900,1000,1023,1024,1025,1200,1536,2047,2048,2049,3072,4095,4096,4097,6144,8192" 2>&1 | grep '"kind"' | sed 's/^{/{"stack": "lumi", /' | tee -a raw/run_$J.jsonl
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $E; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VP\"); sys.path.insert(0,\"$D\"); sys.argv=[\"$D/n1pre.py\",\"2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,31,32,33,36,40,44,48,52,56,60,63,64,65,72,80,96,112,127,128,129,160,192,224,255,256,257,300,320,384,400,448,511,512,513,600,640,700,768,800,900,1000,1023,1024,1025,1200,1536,2047,2048,2049,3072,4095,4096,4097,6144,8192\"]; runpy.run_path(\"$D/n1pre.py\",run_name=\"__main__\")'" 2>&1 | grep '"kind"' | sed 's/^{/{"stack": "vllm", /' | tee -a raw/run_$J.jsonl
echo "## prebuilt $(date +%T)"
CASES=$(python3 -c "print(','.join(f'{t}:{r}' for t in '2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,31,32,33,36,40,44,48,52,56,60,63,64,65,72,80,96,112,127,128,129,160,192,224,255,256,257,300,320,384,400,448,511,512,513,600,640,700,768,800,900,1000,1023,1024,1025,1200,1536,2047,2048,2049,3072,4095,4096,4097,6144,8192'.split(',') for r in ('uniform','zipf1','zipf2')))")
A="--set r4 --rounds 2 --rep 30"
echo "## 1 lumi $(date +%T)"; lu --arms ship,A_u20k@257-511 $A --cases $CASES | tee -a raw/run_$J.jsonl
echo "## 1 vllm $(date +%T)"; vl --arms ship $A --cases $CASES | tee -a raw/run_$J.jsonl
echo "## 2 vllm $(date +%T)"; vl --arms ship $A --cases $CASES --reverse | tee -a raw/run_$J.jsonl
echo "## 2 lumi $(date +%T)"; lu --arms ship,A_u20k@257-511 $A --cases $CASES --reverse | tee -a raw/run_$J.jsonl
ST=1,2,3,5,7,8,15,16,17,31,33,63,64,65,100,128,255,256,257,511,512,513,1000,1024,1025,2048,4096,8192
echo "## self-test lumi $(date +%T)"; $TL "export TILELANG_CACHE_DIR=$CL TRITON_CACHE_DIR=$CL/triton; cd $D; python moe_fused.py --tokens $ST" 2>&1 | grep -v "INFO\|Warning\|in memory cache" | tee -a raw/selftest_lumi_$J.jsonl
echo "## self-test image $(date +%T)"; singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C TRITON_CACHE_DIR=$C/triton HF_HUB_OFFLINE=1 PYTHONPATH=$VP; cd $D; python3 moe_fused.py --tokens $ST" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt\|in memory cache" | tee -a raw/selftest_image_$J.jsonl
date
