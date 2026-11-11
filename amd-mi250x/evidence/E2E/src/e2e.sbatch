#!/bin/bash
#SBATCH --job-name=e2e-vllm
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=02:55:00
# E2E: tokens/s and energy per token of a served 27B (public base, BF16, TP=2 on 2 GCDs, vLLM 0.30 ROCm image), vendor kernels against ours, in ONE allocation.
#   sbatch -A <project> -o <dir>/%x-%j.out --export=ALL,PROJ=<project>,ARMS="s_rocm igs128_rocm s_tri igs128_tri s_rocm ..." e2e.sbatch
# Arms (same serving flags as track V1; a name may repeat, each occurrence is one repetition with a fresh server start):
#   s_rocm       stock vLLM, default attention backend (ROCM_ATTN), no plugin on the path                       = arm A
#   igs128_rocm  same flags + overlay: our gqa_decode recorded into the FULL decode graph (L 65536, skip, S 128) = arm B
#   s_tri        stock vLLM + --attention-backend TRITON_ATTN (a stock flag), no plugin                          = control T
#   igs128_tri   same as s_tri + overlay                                                                           = arm Bt
# The host sampler (e2e_power.py) runs for the whole job; the client writes window edges; e2e_analyse.py integrates.
set -uo pipefail
PROJ=${PROJ:?set PROJ}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
E=${E2E_DIR:-$TL/exp/E2E}; VI=${VI:-$E/vi}; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
MODEL=${MODEL:-/flash/$PROJ/models/Qwen3.8-27B}
ARMS=${ARMS:-s_rocm igs128_rocm s_tri igs128_tri s_rocm igs128_rocm s_tri igs128_tri s_rocm igs128_rocm s_tri igs128_tri}; UP_LIMIT=${UP_LIMIT:-900}; CLIENT_LIMIT=${CLIENT_LIMIT:-1200}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; PORT=$((20000 + J % 20000))
SITE=$TL/venv/lib/python3.12/site-packages
export HLD_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl
export HLD_PRELOAD=${HLD_PRELOAD:-1:1024,8:1024,32:1024,1:2048,8:2048,32:2048,1:8192,8:8192,1:65536,2:65536,4:65536,8:65536,16:65536,32:65536,64:65536} HLD_VERIFY=0 HLD_VERIFY_PAD=0 HLD_LAZY=0 HLD_LAYOUTS=${HLD_LAYOUTS:-rocm_hybrid,lbnhc} HLD_SKIP_LIST=${HLD_SKIP_LIST:-0,128}
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_VLLM_NO_USAGE_STATS=1 SINGULARITYENV_DO_NOT_TRACK=1 SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
stop() { [ -n "${SP:-}" ] && kill -- -"$SP" 2>/dev/null; sleep 8; [ -n "${SP:-}" ] && kill -9 -- -"$SP" 2>/dev/null; for p in $(pgrep -u "$USER" -f "vllm serve|VLLM::|e2e_client"); do grep -q "job_$J/" /proc/$p/cgroup 2>/dev/null && kill -9 $p; done; SP=; sleep 6; }
fin() { stop; [ -n "${PW:-}" ] && kill $PW 2>/dev/null; }
trap fin EXIT
# On a whole node (8 GCDs) set E2E_USE=0:1: the server then sees two GCDs only; the sampler still records every GCD and e2e_analyse.py keeps the busy ones.
[ -n "${E2E_USE:-}" ] && export ROCR_VISIBLE_DEVICES=${E2E_USE//:/,} SINGULARITYENV_ROCR_VISIBLE_DEVICES=${E2E_USE//:/,}   # E2E_USE=0:1 (sbatch --export splits on commas)
echo "## ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
python3 $VI/e2e_power.py $O/power.csv $O/power-meta.json 5 > $O/power.log 2>&1 &
PW=$!
{ echo "## job $J start $(date +%T) arms $ARMS"; md5sum $HLD_SCRIPTS/gqa_decode.py $VI/overlay/horizon_lumi_decode/__init__.py $VI/e2e_client.py $VI/e2e_power.py $VI/e2e.sbatch $VI/precompile.py | sed "s# .*/# #"; } | tee $O/header.txt
echo "## compile start $(date +%T)"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/precompile.py $i 12 2" > $O/compile-$i.log 2>&1 &
done; wait $(jobs -p | grep -v "^$PW$"); echo "## compile end $(date +%T)"; grep -h '^compiled' $O/compile-*.log | wc -l; grep -h 'Traceback\|Error' $O/compile-*.log | cut -c1-200 | sort | uniq -c | tail -10
sleep 20; t0=$(date +%s.%N); sleep 30; echo "{\"kind\": \"idle_bare\", \"t_start\": $t0, \"t_end\": $(date +%s.%N)}" > $O/idle-bare.jsonl   # GCDs with no process at all
declare -A REP
for a in $ARMS; do
  r=${REP[$a]:-0}; REP[$a]=$((r + 1)); T=$a-r$r
  echo "## arm $a rep $r start $(date +%T)"
  C=$E/cache/$J/$a; mkdir -p $C    # caches are per job: two jobs never write the same compile cache
  FL=""; [[ $a == *_tri ]] && FL="--attention-backend TRITON_ATTN"
  [[ $a == ig* ]] && export SINGULARITYENV_HLD_CAPTURE=1 || unset SINGULARITYENV_HLD_CAPTURE
  S=0; [[ $a =~ ^igs([0-9]+)_ ]] && S=${BASH_REMATCH[1]}; export SINGULARITYENV_HLD_SKIP_S=$S
  if [[ $a == i* ]]; then export SINGULARITYENV_HLD_ENABLE=1 SINGULARITYENV_PYTHONPATH=$VI/overlay:$SITE; else unset SINGULARITYENV_HLD_ENABLE SINGULARITYENV_PYTHONPATH; fi
  mkdir -p $O/ctr-$T; export SINGULARITYENV_HLD_COUNTER_DIR=$O/ctr-$T
  export SINGULARITYENV_XDG_CACHE_HOME=$C SINGULARITYENV_VLLM_CACHE_ROOT=$C/vllm SINGULARITYENV_TRITON_CACHE_DIR=$C/triton
  setsid singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL --served-model-name e2e --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88 \
     --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 127.0.0.1 --port $PORT $FL > $O/serve-$T.log 2>&1 &
  SP=$!; t0=$(date +%s); up=0
  for ((w=0; w<UP_LIMIT; w+=10)); do
    singularity exec $SIF python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SP 2>/dev/null || break; sleep 10
  done
  echo "## arm $a rep $r up=$up after $(( $(date +%s)-t0 )) s"
  if [ $up = 1 ]; then
    timeout $CLIENT_LIMIT singularity exec -B /pfs,/scratch,/flash $SIF python3 $VI/e2e_client.py $PORT e2e $a $r $O > $O/client-$T.jsonl 2> $O/client-$T.err
    echo "## arm $a rep $r client rc=$? rows $(grep -c '"kind": "cell"' $O/client-$T.jsonl) $(date +%T)"; tail -2 $O/client-$T.err | cut -c1-300
  fi
  stop
  grep -hE 'HLD ' $O/serve-$T.log | grep -E '"ev": "(counters|error|lazy_compile_start)' | cut -c1-300 | tail -4; head -c 300 $O/ctr-$T/*.json 2>/dev/null | cut -c1-600; echo
  grep -h "falling back\|GPU KV cache size\|Using .* backend\|attention backend" $O/serve-$T.log | sed 's/(.*pid=[0-9]*) //' | cut -c1-220 | sort | uniq -c | head -8
  echo "## arm $a rep $r end $(date +%T)"
done
echo "## job end $(date +%T)"
