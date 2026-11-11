#!/bin/bash
#SBATCH --job-name=g2-vllm
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=14
#SBATCH --mem=120G
#SBATCH --time=02:20:00
# G2 (g1.sbatch with the landed tile table, a TRITON_ATTN baseline and the equivalence set): time to first token of fresh prompts on the served 27B (public base, BF16, TP=2 on 2 GCDs, vLLM 0.30 ROCm image), vLLM's dense-linear path (hipBLASLt) against ours (gemm_bf16 through horizon_lumi_gemm), ONE allocation.
#   sbatch -A <project> -o <dir>/%x-%j.out --export=ALL,PROJ=<project>,ARMS="gv_rocm s_rocm g_rocm g_rocm s_rocm" g1.sbatch
# Arms (p1.sbatch's serving flags; a name may repeat, each occurrence is a fresh server start). Letters before _:
#   s stock, no plugin on the path     g + horizon_lumi_gemm     gv the same with HLG_VERIFY=2 (the first 2 calls per kernel also run vLLM's path and log the difference; not a timing arm)
#   f + horizon_lumi_prefill (fg_rocm = both plugins)          suffix _tri instead of _rocm adds the stock flag --attention-backend TRITON_ATTN to that arm (s_tri = stock with that flag)
# MNBT=<n> adds --max-num-batched-tokens n to EVERY arm. The GEMM kernels are those of gemm_bf16.ROWS_FORMS (plugin version 3); HLG_MS / HLG_FORMS override.
# Per arm: p1_client.py (time to first token cells), then eq_client.py (12 fixed greedy prompts with logprobs, compared against stock afterwards).
set -uo pipefail
PROJ=${PROJ:?set PROJ}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
E=${G1_DIR:-$TL/exp/G1}; VI=${VI:-$E/vi3}; SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
MODEL=${MODEL:-/flash/$PROJ/models/Qwen3.8-27B}
ARMS=${ARMS:-gv_rocm s_rocm g_rocm g_rocm s_rocm}; UP_LIMIT=${UP_LIMIT:-900}; CLIENT_LIMIT=${CLIENT_LIMIT:-1200}; MNBT=${MNBT:-}
J=$SLURM_JOB_ID; O=$E/raw/$J; mkdir -p $O; PORT=$((20000 + J % 20000))
SITE=$TL/venv/lib/python3.12/site-packages
HLG_MS=${HLG_MS:-}
export HLG_MS HLG_SCRIPTS=$VI/gm-src HLP_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl HLG_SHAPES=${HLG_SHAPES:-17408x5120,5120x8704,8192x5120,7168x5120,5120x3072} HLG_MIN=${HLG_MIN:-512} HLG_MAXPAD=${HLG_MAXPAD:-0} HLG_FORMS="${HLG_FORMS:-}"
export HLP_LADDER=${HLP_LADDER:-1024:16384:0.06} HLP_PRELOAD=1 HLP_LAZY=0 HLP_GUARD=1 HLP_BUCKETS_784=1 HLP_CTX_MAX=${HLP_CTX_MAX:-9408}
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_VLLM_NO_USAGE_STATS=1 SINGULARITYENV_DO_NOT_TRACK=1 SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn
stop() { [ -n "${SP:-}" ] && kill -- -"$SP" 2>/dev/null; sleep 8; [ -n "${SP:-}" ] && kill -9 -- -"$SP" 2>/dev/null; for p in $(pgrep -u "$USER" -f "vllm serve|VLLM::|p1_client|eq_client"); do grep -q "job_$J/" /proc/$p/cgroup 2>/dev/null && kill -9 $p; done; SP=; sleep 6; }
fin() { stop; [ -n "${PW:-}" ] && kill $PW 2>/dev/null; }
trap fin EXIT
# On a whole node (8 GCDs) set E2E_USE=0:1: the server then sees two GCDs only (the two dies of one card, no other tenant on the card); the sampler still records every GCD.
[ -n "${E2E_USE:-}" ] && export ROCR_VISIBLE_DEVICES=${E2E_USE//:/,} SINGULARITYENV_ROCR_VISIBLE_DEVICES=${E2E_USE//:/,}
echo "## ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset} node $(hostname)"
python3 $VI/e2e_power.py $O/power.csv $O/power-meta.json 5 > $O/power.log 2>&1 &
PW=$!
{ echo "## job $J start $(date +%T) arms $ARMS MNBT=${MNBT:-default} HLG_MS=$HLG_MS HLG_MIN=$HLG_MIN HLG_MAXPAD=$HLG_MAXPAD"; md5sum $VI/gm-src/gemm_prefetch.py $VI/gm-src/gemm_bf16.py $VI/hld-src/gqa_fa.py $VI/overlay/horizon_lumi_gemm/__init__.py $VI/overlay/horizon_lumi_prefill/__init__.py $VI/p1_client.py $VI/eq_client.py $VI/e2e_power.py $VI/g2.sbatch $VI/g1_precompile.py $VI/p1_precompile.py | sed "s# .*/# #"; } | tee $O/header.txt
echo "## compile start $(date +%T)"
for i in 0 1 2 3 4 5 6 7 8 9 10 11; do
  singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 PYTHONPATH=$VI/overlay:$SITE; python3 $VI/g1_precompile.py $i 12 && { [[ ' $ARMS' != *' f'* && ' $ARMS' != *' c'* ]] || python3 $VI/p1_precompile.py $i 12 2; }" > $O/compile-$i.log 2>&1 &
done; wait $(jobs -p | grep -v "^$PW$"); echo "## compile end $(date +%T) compiled $(grep -h '^compiled' $O/compile-*.log | wc -l)"; grep -h 'Traceback\|Error' $O/compile-*.log | cut -c1-200 | sort | uniq -c | tail -10
declare -A REP
for a in $ARMS; do
  r=${REP[$a]:-0}; REP[$a]=$((r + 1)); T=$a-r$r
  echo "## arm $a rep $r start $(date +%T)"
  C=$E/cache/$J/$a; mkdir -p $C $O/ctr-$T
  K=${a%%_*}; FL=(); [[ $a == *_tri ]] && FL+=(--attention-backend TRITON_ATTN)
  unset SINGULARITYENV_HLG_ENABLE SINGULARITYENV_HLP_ENABLE SINGULARITYENV_PYTHONPATH SINGULARITYENV_HLG_VERIFY SINGULARITYENV_HLP_VERIFY
  [[ $K == *g* ]] && export SINGULARITYENV_HLG_ENABLE=1 SINGULARITYENV_PYTHONPATH=$VI/overlay:$SITE SINGULARITYENV_HLG_VERIFY=0
  [[ $K == *f* ]] && export SINGULARITYENV_HLP_ENABLE=1 SINGULARITYENV_PYTHONPATH=$VI/overlay:$SITE SINGULARITYENV_HLP_VERIFY=0
  [[ $K == *gv* ]] && export SINGULARITYENV_HLG_VERIFY=2
  unset SINGULARITYENV_HLP_CTX      # c: horizon_lumi_prefill with HLP_CTX=1 (also chunks with cached context, up to HLP_CTX_MAX tokens); cg_rocm = that + the GEMM plugin
  [[ $K == *c* ]] && export SINGULARITYENV_HLP_ENABLE=1 SINGULARITYENV_PYTHONPATH=$VI/overlay:$SITE SINGULARITYENV_HLP_VERIFY=0 SINGULARITYENV_HLP_CTX=1
  [ -n "$MNBT" ] && FL+=(--max-num-batched-tokens $MNBT)
  export SINGULARITYENV_HLG_COUNTER_DIR=$O/ctr-$T SINGULARITYENV_HLP_COUNTER_DIR=$O/ctr-$T
  export SINGULARITYENV_XDG_CACHE_HOME=$C SINGULARITYENV_VLLM_CACHE_ROOT=$C/vllm SINGULARITYENV_TRITON_CACHE_DIR=$C/triton
  setsid singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL --served-model-name p1 --dtype bfloat16 --tensor-parallel-size 2 --max-model-len 65536 --gpu-memory-utilization 0.88 \
     --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 127.0.0.1 --port $PORT "${FL[@]}" > $O/serve-$T.log 2>&1 &
  SP=$!; t0=$(date +%s); up=0
  for ((w=0; w<UP_LIMIT; w+=10)); do
    singularity exec $SIF python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$PORT/v1/models', timeout=5)" >/dev/null 2>&1 && { up=1; break; }
    kill -0 $SP 2>/dev/null || break; sleep 10
  done
  echo "## arm $a rep $r up=$up after $(( $(date +%s)-t0 )) s"
  if [ $up = 1 ]; then
    timeout $CLIENT_LIMIT singularity exec -B /pfs,/scratch,/flash $SIF python3 $VI/p1_client.py $PORT p1 $a $r $O > $O/client-$T.jsonl 2> $O/client-$T.err
    echo "## arm $a rep $r client rc=$? rows $(grep -c '"kind": "cell"' $O/client-$T.jsonl) $(date +%T)"; tail -2 $O/client-$T.err | cut -c1-300
    timeout 600 singularity exec -B /pfs,/scratch,/flash $SIF python3 $VI/eq_client.py $PORT p1 $a $r $O > $O/eqrun-$T.log 2>&1
    echo "## arm $a rep $r eq rc=$? rows $(wc -l < $O/eq-$T.jsonl) $(date +%T)"
  else tail -5 $O/serve-$T.log | cut -c1-300; fi
  stop
  grep -hE 'HL[GP] ' $O/serve-$T.log | grep -E '"ev": "(counters|error|load_error)' | cut -c1-400 | tail -6; echo
  grep -h 'HLG ' $O/serve-$T.log | grep '"ev": "verify"' | cut -c1-260 | tail -6
  grep -h "GPU KV cache size" $O/serve-$T.log | sed 's/(.*pid=[0-9]*) //' | cut -c1-160 | head -2
  echo "## arm $a rep $r end $(date +%T)"
done
rm -rf $E/cache/$J
echo "## job end $(date +%T)"
