#!/bin/bash
#SBATCH --job-name=v1-vllm
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=01:20:00
#SBATCH --output=%x-%j.out
# V1: stock vLLM 0.30 against vLLM + landed decode, 27B public base, TP=2, 2 GCDs, in one allocation, in the order of ARMS.
#   ARMS="s_rocm ig_rocm p_rocm ip_rocm" ROUNDS=2 sbatch v1.sbatch. Prefix: s_ stock default (FULL_AND_PIECEWISE); i_ plugin, default mode, hook off in the FULL graph, so stock on decode steps; ig_ plugin recorded into the FULL graph; igsN_ same with the uniform dead="skip" form of N splits at L 65536; p_ stock forced PIECEWISE; ip_ plugin, forced PIECEWISE, hook on every decode step. _tri adds --attention-backend TRITON_ATTN. s_ and p_ have no plugin on the path at all.
# Run from the batch shell, no srun, as serve.sbatch does. Caches are this track's own, nothing shared is written.
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
E=${V1_DIR:-$TL/exp/V1}; VI=${VI:-$E/vi}; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
MODEL=${MODEL:-/flash/$PROJ/models/Qwen3.8-27B}; TEMPLATE=${CHAT_TEMPLATE:-}; SERVE_CHECK=${SERVE_CHECK:-$VI/serve_check.py}
TPL=""; [ -n "$TEMPLATE" ] && TPL="--chat-template $TEMPLATE"   # optional: the model's own template is used when empty
ARMS=${ARMS:-s_rocm i_rocm s_tri i_tri}; ROUNDS=${ROUNDS:-3}; UP_LIMIT=${UP_LIMIT:-900}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; PORT=$((20000 + J % 20000))
SITE=$TL/venv/lib/python3.12/site-packages
export HLD_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl
export HLD_PRELOAD=${HLD_PRELOAD:-1:1024,8:1024,32:1024,1:2048,8:2048,32:2048,1:8192,8:8192,1:65536,2:65536,4:65536,8:65536,16:65536,32:65536,64:65536} HLD_VERIFY=${HLD_VERIFY:-4} HLD_VERIFY_PAD=${HLD_VERIFY_PAD:-4} HLD_LAZY=${HLD_LAZY:-0} HLD_LAYOUTS=${HLD_LAYOUTS:-rocm_hybrid,lbnhc}
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_VLLM_NO_USAGE_STATS=1 SINGULARITYENV_DO_NOT_TRACK=1 SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
stop() { [ -n "${SP:-}" ] && kill -- -"$SP" 2>/dev/null; sleep 8; [ -n "${SP:-}" ] && kill -9 -- -"$SP" 2>/dev/null; for p in $(pgrep -u "$USER" -f "vllm serve|VLLM::|v1_client"); do grep -q "job_$J/" /proc/$p/cgroup 2>/dev/null && kill -9 $p; done; SP=; sleep 6; }
trap stop EXIT
{ echo "## job $J start $(date +%T) arms $ARMS rounds $ROUNDS"; md5sum $HLD_SCRIPTS/gqa_decode.py $VI/overlay/horizon_lumi_decode/__init__.py $VI/v1_client.py; cat $SLURM_SUBMIT_DIR/.none 2>/dev/null; } | tee $O/header.txt
if [[ " $ARMS" == *" i"* ]]; then
  echo "## compile start $(date +%T)"
  for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
    singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; HLD_SKIP_LIST=${HLD_SKIP_LIST:-0} python3 $VI/precompile.py $i 12 2" > $O/compile-$i.log 2>&1 &
  done; wait; echo "## compile end $(date +%T)"; grep -h '^compiled\|Traceback\|Error' $O/compile-*.log | cut -c1-200 | sort | tail -40
fi
for a in $ARMS; do
  echo "## arm $a start $(date +%T)"
  C=$E/cache/$a; mkdir -p $C
  FL=""; [[ $a == *_tri ]] && FL="--attention-backend TRITON_ATTN"
  [[ $a == p_* || $a == ip_* ]] && FL="$FL --compilation-config {\"cudagraph_mode\":\"PIECEWISE\"}"
  [[ $a == ig* ]] && export SINGULARITYENV_HLD_CAPTURE=1 || unset SINGULARITYENV_HLD_CAPTURE
  S=0; [[ $a =~ ^igs([0-9]+)_ ]] && S=${BASH_REMATCH[1]}; export SINGULARITYENV_HLD_SKIP_S=$S
  if [[ $a == i* ]]; then export SINGULARITYENV_HLD_ENABLE=1 SINGULARITYENV_PYTHONPATH=$VI/overlay:$SITE; else unset SINGULARITYENV_HLD_ENABLE SINGULARITYENV_PYTHONPATH; fi
  mkdir -p $O/ctr-$a; export SINGULARITYENV_HLD_COUNTER_DIR=$O/ctr-$a
  export SINGULARITYENV_XDG_CACHE_HOME=$C SINGULARITYENV_VLLM_CACHE_ROOT=$C/vllm SINGULARITYENV_TRITON_CACHE_DIR=$C/triton
  setsid singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL --served-model-name v1 --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88 \
     --limit-mm-per-prompt '{"image":0,"video":0}' $TPL --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 127.0.0.1 --port $PORT $FL > $O/serve-$a.log 2>&1 &
  SP=$!; t0=$(date +%s); up=0
  for ((w=0; w<UP_LIMIT; w+=10)); do
    singularity exec $SIF python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SP 2>/dev/null || break; sleep 10
  done
  echo "## arm $a up=$up after $(( $(date +%s)-t0 )) s"
  if [ $up = 1 ]; then
    timeout 1500 singularity exec -B /pfs,/scratch,/flash $SIF python3 $VI/v1_client.py $PORT v1 $ROUNDS $a $O 2>&1 | tee -a $O/client-$a.jsonl | grep -c decode
    echo "## arm $a client rc=${PIPESTATUS[0]} $(date +%T)"
    timeout 900 singularity exec -B /pfs,/scratch,/flash $SIF python3 $SERVE_CHECK $PORT v1 --needle > $O/servecheck-$a.txt 2>&1
    echo "## arm $a serve_check rc=$? $(date +%T)"; tail -1 $O/servecheck-$a.txt | cut -c1-120
  fi
  stop
  grep -hE 'HLD ' $O/serve-$a.log | grep -E '"ev": "(counters|error|lazy_compile_start)' | cut -c1-300 | tail -8; head -c 400 $O/ctr-$a/*.json 2>/dev/null | cut -c1-400
  echo "## arm $a end $(date +%T)"
done
echo "## job end $(date +%T)"
