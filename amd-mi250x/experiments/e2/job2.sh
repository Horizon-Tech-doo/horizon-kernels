#!/bin/bash
# forced unroll through hipcc's HIPCC_COMPILE_FLAGS_APPEND (compile_flags is not used by the HIP path in TileLang 0.1.15); separate caches per flag
E=/scratch/project_XXXXXXXXX/tilelang/exp/E2; cd $E
SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
ONLY="--only base,F1,F3_dump,F5_mg_dump,F7_hoist"
lumi(){ /scratch/project_XXXXXXXXX/tilelang/scripts/tl_run.sh "export TILELANG_CACHE_DIR=$E/cache-u$1 TRITON_CACHE_DIR=$E/cache-u$1/triton HIPCC_COMPILE_FLAGS_APPEND='-mllvm -unroll-threshold=$1'; cd $E; python -I e2run.py --data $E/data.pt --stack lumi --rounds 1 --rep 50 $ONLY" 2>&1 | grep -v "INFO\|Warning" | sed -u "s/^{/{\"unroll\": $1, /"; }
vl(){ singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache-vu$1 TRITON_CACHE_DIR=$E/cache-vu$1/triton HF_HUB_OFFLINE=1 HIPCC_COMPILE_FLAGS_APPEND='-mllvm -unroll-threshold=$1' PYTHONPATH=/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages; cd $E; python3 -I -c 'import sys; sys.path.insert(0,\"/scratch/project_XXXXXXXXX/tilelang/venv/lib/python3.12/site-packages\"); sys.path.insert(0,\"$E\"); import runpy; runpy.run_path(\"$E/e2run.py\", run_name=\"__main__\")' --data $E/data.pt --stack vllm --rounds 1 --rep 50 $ONLY" 2>&1 | grep -v "INFO\|Warning\|amdsmi\|raw_cnt" | sed -u "s/^{/{\"unroll\": $1, /"; }
for c in 1 2 3; do for u in 2000 20000; do
 echo "## $c lumi u=$u $(date +%T)"; lumi $u | tee -a raw/runu_$SLURM_JOB_ID.jsonl
done; echo "## $c vllm u=20000 $(date +%T)"; vl 20000 | tee -a raw/runu_$SLURM_JOB_ID.jsonl; done
