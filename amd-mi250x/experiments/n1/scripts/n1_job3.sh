#!/bin/bash
# one allocation, cycles of LUMI stack then vLLM image. $1 cycles, $2.. args to n1run. HIPCC flags are set inside moe_fused.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N1; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
[ -f data.pt ] || /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd $E; python -I n1_mk.py $E/data.pt" 2>&1 | grep -v Warning
[ -n "$MK2" ] && /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd $E; python -I n1_mk2.py $E/data.pt" 2>&1 | grep -v Warning
DIR=${DIR:-$E}; R=$1; shift; IMAGE_TL=${IMAGE_TL:-1}
lumi(){ /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python -I $DIR/n1run.py --data $E/data.pt --stack lumi $*" 2>&1 | grep -v "INFO\|Warning" ; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages\"); sys.path.insert(0,\"$DIR\"); import runpy; runpy.run_path(\"$DIR/n1run.py\", run_name=\"__main__\")' --data $E/data.pt --stack vllm $* $VL_EXTRA" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" ; }
for c in $(seq 1 $R); do
  echo "## cycle $c lumi $(date +%T)"; lumi "$@" | tee -a raw/run_$SLURM_JOB_ID.jsonl
  echo "## cycle $c vllm $(date +%T)"; vl "$@" | tee -a raw/run_$SLURM_JOB_ID.jsonl
done
date
