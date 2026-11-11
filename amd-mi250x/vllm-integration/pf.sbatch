#!/bin/bash
#SBATCH --job-name=pf-vllm
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=02:30:00
# PROFILE: where a serving step's wall time goes on the P1/E2E server (public 27B, BF16, TP=2 on 2 GCDs, vLLM 0.30 ROCm image), stock kernels only, in ONE allocation.
#   sbatch -A <project> -o <dir>/%x-%j.out --export=ALL,PROJ=<project>,ARMS="s_d ps_d py_d pyn_d po_d po_m16 ..." pf.sbatch
# Arm = <kind>_<d|mN>[_capW]; each occurrence is a fresh server start. Kind:
#   s    stock, no plugin on the path (client-side numbers only)
#   sk   stock with VLLM_ROCM_USE_SKINNY_GEMM=0 (every linear through torch F.linear = hipBLASLt)
#   ps   + horizon_lumi_profile HPF_MODE=step (clock reads around the runner's step, no synchronisation)
#   py   + HPF_MODE=sync (one synchronise per step)
#   pyn  as py with --compilation-config '{"cudagraph_mode":"NONE"}' (the cost of not replaying graphs, without per-operator timers)
#   po   HPF_MODE=ops with cudagraph NONE: the per-operator breakdown
# _d = vLLM's default max_num_batched_tokens (2048: longer prompts are prefilled in chunks); _mN = --max-num-batched-tokens N*1024.
set -uo pipefail
PROJ=${PROJ:?set PROJ}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
E=${PF_DIR:-$TL/exp/PROFILE}; VI=${VI:-$E/vi}; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
MODEL=${MODEL:-/flash/$PROJ/models/Qwen3.8-27B}
ARMS=${ARMS:-s_d ps_d py_d pyn_d po_d po_m16 py_m16 s_m16}; UP_LIMIT=${UP_LIMIT:-900}; CLIENT_LIMIT=${CLIENT_LIMIT:-1500}; CELLS=${CELLS:-}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; PORT=$((20000 + J % 20000))
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_VLLM_NO_USAGE_STATS=1 SINGULARITYENV_DO_NOT_TRACK=1 SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
stop() { [ -n "${SP:-}" ] && kill -- -"$SP" 2>/dev/null; sleep 8; [ -n "${SP:-}" ] && kill -9 -- -"$SP" 2>/dev/null; for p in $(pgrep -u "$USER" -f "vllm serve|VLLM::|pf_client"); do grep -q "job_$J/" /proc/$p/cgroup 2>/dev/null && kill -9 $p; done; SP=; sleep 6; }
fin() { stop; [ -n "${PW:-}" ] && kill $PW 2>/dev/null; }
trap fin EXIT
python3 $VI/e2e_power.py $O/power.csv $O/power-meta.json 5 > $O/power.log 2>&1 &
PW=$!
{ echo "## job $J start $(date +%T) arms $ARMS cells ${CELLS:-default}"; md5sum $VI/overlay/horizon_lumi_profile/__init__.py $VI/pf_client.py $VI/e2e_power.py $VI/pf.sbatch | sed "s# .*/# #"; } | tee $O/header.txt
declare -A REP
for a in $ARMS; do
  r=${REP[$a]:-0}; REP[$a]=$((r + 1)); T=$a-r$r
  echo "## arm $a rep $r start $(date +%T)"
  C=$E/cache/$J/$a; mkdir -p $C $O/hpf-$T
  K=${a%%_*}; R=${a#*_}; FL=()
  unset SINGULARITYENV_HPF_ENABLE SINGULARITYENV_PYTHONPATH SINGULARITYENV_HPF_MODE SINGULARITYENV_VLLM_ROCM_USE_SKINNY_GEMM
  case $K in
    s) ;;
    sk) export SINGULARITYENV_VLLM_ROCM_USE_SKINNY_GEMM=0 ;;      # stock, no plugin, vLLM's own small-row kernels (wvSplitK, LLMM1) switched off by vLLM's own environment switch
    ps) export SINGULARITYENV_HPF_MODE=step ;;
    py) export SINGULARITYENV_HPF_MODE=sync ;;
    pyn) export SINGULARITYENV_HPF_MODE=sync; FL+=(--compilation-config '{"cudagraph_mode":"NONE"}') ;;
    po) export SINGULARITYENV_HPF_MODE=ops; FL+=(--compilation-config '{"cudagraph_mode":"NONE"}') ;;
    *) echo "unknown arm $a"; continue ;;
  esac
  [ $K != s ] && [ $K != sk ] && export SINGULARITYENV_HPF_ENABLE=1 SINGULARITYENV_PYTHONPATH=$VI/overlay SINGULARITYENV_HPF_DIR=$O/hpf-$T
  [[ $R =~ ^m([0-9]+) ]] && FL+=(--max-num-batched-tokens $((BASH_REMATCH[1] * 1024)))
  export SINGULARITYENV_XDG_CACHE_HOME=$C SINGULARITYENV_VLLM_CACHE_ROOT=$C/vllm SINGULARITYENV_TRITON_CACHE_DIR=$C/triton
  setsid singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL --served-model-name pf --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88 \
     --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 127.0.0.1 --port $PORT "${FL[@]}" > $O/serve-$T.log 2>&1 &
  SP=$!; t0=$(date +%s); up=0
  for ((w=0; w<UP_LIMIT; w+=10)); do
    singularity exec $SIF python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SP 2>/dev/null || break; sleep 10
  done
  echo "## arm $a rep $r up=$up after $(( $(date +%s)-t0 )) s flags ${FL[*]:-none}"
  if [ $up = 1 ]; then
    timeout $CLIENT_LIMIT singularity exec -B /pfs,/scratch,/flash $SIF python3 $VI/pf_client.py $PORT pf $a $r $O $CELLS > $O/client-$T.jsonl 2> $O/client-$T.err
    echo "## arm $a rep $r client rc=$? rows $(grep -c '"kind": "cell"' $O/client-$T.jsonl) $(date +%T)"; tail -2 $O/client-$T.err | cut -c1-300
  else tail -5 $O/serve-$T.log | cut -c1-400; fi
  stop
  grep -h 'HPF ' $O/serve-$T.log | cut -c1-600 | tail -4; wc -l $O/hpf-$T/*.jsonl 2>/dev/null | tail -3
  grep -h "Traceback\|Error\|falling back\|GPU KV cache size\|Using .* backend\|max_num_batched_tokens\|cudagraph" $O/serve-$T.log | sed 's/(.*pid=[0-9]*) //' | cut -c1-220 | sort | uniq -c | sort -rn | head -10
  echo "## arm $a rep $r end $(date +%T)"
done
rm -rf $E/cache/$J
echo "## job end $(date +%T)"
