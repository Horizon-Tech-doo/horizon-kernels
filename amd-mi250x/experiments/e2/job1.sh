#!/bin/bash
# one allocation: preflight (LUMI stack then vLLM stack, one rep each kind)
E=/scratch/project_XXXXXXXXX/tilelang/exp/E2; cd $E; mkdir -p raw
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
[ -f data.pt ] || /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "cd $E; python -I mk_data.py $E/data.pt" 2>&1 | grep -v Warning
date; rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
R=$1; ROUNDS=${2:-1}; REP=${3:-30}
lumi(){ /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache TRITON_CACHE_DIR=$E/cache/triton; cd $E; python -I e2run.py --data $E/data.pt --stack lumi --rounds $ROUNDS --rep $REP $1" 2>&1 | grep -v "INFO\|Warning" ; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-v TRITON_CACHE_DIR=$E/cache-v/triton HF_HUB_OFFLINE=1 PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages\"); sys.argv=[\"e2run.py\"]+sys.argv[1:]; sys.path.insert(0,\"$E\"); import runpy; runpy.run_path(\"$E/e2run.py\", run_name=\"__main__\")' --data $E/data.pt --stack vllm --rounds $ROUNDS --rep $REP $1" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" ; }
for c in $(seq 1 $R); do
  echo "## cycle $c lumi $(date +%T)"; lumi "$4" | tee -a raw/run_$SLURM_JOB_ID.jsonl
  echo "## cycle $c vllm $(date +%T)"; vl "$4" | tee -a raw/run_$SLURM_JOB_ID.jsonl
done
date; rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
