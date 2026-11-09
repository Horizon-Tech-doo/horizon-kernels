#!/bin/bash
#SBATCH --job-name=vllm-mn
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Track 5: vLLM TP=4 on one node (4 GCDs) and on two nodes (2 GCDs each), inside one allocation, in the order of ARMS.
#   ARMS="one two" sbatch vllm_mn.sbatch
# Before the model loads each arm runs a short RCCL test in the image, so a transport failure shows up as itself.
# Every step names its node: on this system --relative did not put a step on the node it was expected on (observed, jobs 23515891, 23516410).
# ROCm 7.2.3 in the image is outside LUMI's supported range. RCCL has no OFI plugin in the image, so sockets are expected.
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
V=${VLLM_DIR:-/scratch/$PROJ/vllm-check}
SC=$TL/scaling
SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
MODEL=${MODEL:-/flash/$PROJ/models/Qwen3.8-27B}
SERVE_CHECK=${SERVE_CHECK:-$SC/serve_check.py}; CHAT_TEMPLATE=${CHAT_TEMPLATE:-}   # template optional: the model's own one is used when empty
ARMS=${ARMS:-one two}; UP_LIMIT=${UP_LIMIT:-900}; ROUNDS=${ROUNDS:-3}
# Separate compile caches per layout: the vLLM cache keys on rank but stores device ordinals (observed, 23516861). CACHE_one, CACHE_two, CACHE_ofi name them under $V/cache.
CACHE_one=${CACHE_one:-mn-tp4}; CACHE_two=${CACHE_two:-mn-tp4}; CACHE_ofi=${CACHE_ofi:-$CACHE_two}
CHECK_LIMIT=${CHECK_LIMIT:-240}; CLIENT_LIMIT=${CLIENT_LIMIT:-360}   # bounds a hang; a timed-out step reports rc 124
OFI=${OFI_DIR:-$SC/ofi}
LOGD=$TL/logs/vllm-mn-$SLURM_JOB_ID; mkdir -p $LOGD
HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST))
ipof='ip -4 -o addr show hsn0 | sed -E "s/.*inet ([0-9.]+).*/\1/"'
setcache() { local C=$V/cache/$1; mkdir -p $C; export SINGULARITYENV_XDG_CACHE_HOME=$C SINGULARITYENV_VLLM_CACHE_ROOT=$C/vllm SINGULARITYENV_TRITON_CACHE_DIR=$C/triton; echo "## cache $1"; }
export SINGULARITYENV_HF_HUB_OFFLINE=1 SINGULARITYENV_PYTHONNOUSERSITE=1 \
       SINGULARITYENV_VLLM_NO_USAGE_STATS=1 SINGULARITYENV_DO_NOT_TRACK=1 SINGULARITYENV_VLLM_WORKER_MULTIPROC_METHOD=spawn \
       SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 SINGULARITYENV_NCCL_NET_GDR_LEVEL=PHB \
       SINGULARITYENV_NCCL_DEBUG=${NCCL_DEBUG_LEVEL:-WARN}
[ -n "${NCCL_SUBSYS:-}" ] && export SINGULARITYENV_NCCL_DEBUG_SUBSYS=$NCCL_SUBSYS
FLAGS="--served-model-name mn --dtype bfloat16 --tensor-parallel-size 4 --max-model-len 65536 --max-num-seqs 64 --gpu-memory-utilization 0.88"
FLAGS="$FLAGS --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder"
[ -n "$CHAT_TEMPLATE" ] && FLAGS="$FLAGS --chat-template $CHAT_TEMPLATE"
LIM='{"image":0,"video":0}'
on0() { srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 --cpus-per-task=2 --cpu-bind=none "$@"; }
PIDS=()
stop() { for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done; sleep 8; for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill -9 "$p" 2>/dev/null; done; PIDS=(); srun --overlap --ntasks=$SLURM_NNODES --ntasks-per-node=1 --cpu-bind=none bash -c 'for p in $(pgrep -u "$USER" -f "vllm serve|VLLM::"); do grep -q "job_'$SLURM_JOB_ID'/" /proc/$p/cgroup 2>/dev/null && kill -9 $p; done' 2>/dev/null; sleep 5; }
trap stop EXIT

comm() { # arm nodes tasks-per-node
  echo "## $1 comm start $(date +%T)"
  MASTER_ADDR=$(srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 bash -c "$ipof") MASTER_PORT=29600 \
  SINGULARITYENV_NCCL_DEBUG=INFO timeout 240 srun --overlap --nodelist=$(IFS=,; echo "${HOSTS[*]:0:$2}") --nodes=$2 --ntasks=$(( $2 * $3 )) --ntasks-per-node=$3 --cpus-per-task=7 --cpu-bind=none \
    bash -c 'export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID SINGULARITYENV_ROCR_VISIBLE_DEVICES=$SLURM_LOCALID; singularity exec -B /pfs,/scratch,/flash '$SIF' python3 '$SC'/vllm_mn_comm.py' > $LOGD/comm-$1.log 2>&1
  echo "## $1 comm rc=$? end $(date +%T)"
  grep -oE '\{"(op|check).*\}' $LOGD/comm-$1.log   # rows can carry a leading space in the merged log
  grep -iE 'NET/|using network|socket|cxi|libfabric|fallback|tcp' $LOGD/comm-$1.log | grep -v alt_rsmi | sed -E 's/^[^ ]+ *//' | sort | uniq -c | sort -rn | head -12 | cut -c1-200
}

arm() { # name
  local a=$1 t0 p=8100 w dead=""
  t0=$(date +%s)
  unset SINGULARITYENV_LD_LIBRARY_PATH SINGULARITYENV_FI_MR_CACHE_MONITOR SINGULARITYENV_FI_CXI_DEFAULT_CQ_SIZE   # also when the previous arm returned early
  eval setcache \$CACHE_$a
  if [ "$a" = ofi ]; then
    # Hand adapter plus LUMI's plugin, libfabric and libcxi from lumi-pytorch-rocm-6.2.4. One step for both nodes: libfabric CXI endpoints in different steps may hold different VNIs.
    export SINGULARITYENV_LD_LIBRARY_PATH=$OFI/adapter:$OFI/shim:$OFI/aws-ofi-rccl:$OFI/libfabric/lib:$OFI/shs-libcxi/lib:/opt/rocm-7.2.3/lib
    export SINGULARITYENV_FI_MR_CACHE_MONITOR=userfaultfd SINGULARITYENV_FI_CXI_DEFAULT_CQ_SIZE=131072
    comm ofi 2 2
    local HEAD; HEAD=$(srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 bash -c "$ipof")
    GP=0,1
    cat > $LOGD/ofi-serve.sh <<OFIEOF
#!/bin/bash
export VLLM_HOST_IP=\$($ipof); export SINGULARITYENV_VLLM_HOST_IP=\$VLLM_HOST_IP ROCR_VISIBLE_DEVICES=$GP SINGULARITYENV_ROCR_VISIBLE_DEVICES=$GP
if [ "\$SLURM_PROCID" = 0 ]; then
  exec singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL $FLAGS --nnodes 2 --node-rank 0 --master-addr $HEAD --master-port 29502 --limit-mm-per-prompt '$LIM' --host 127.0.0.1 --port $p > $LOGD/serve-ofi-r0.log 2>&1
else
  exec singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL $FLAGS --nnodes 2 --node-rank 1 --master-addr $HEAD --master-port 29502 --headless --limit-mm-per-prompt '$LIM' > $LOGD/serve-ofi-r1.log 2>&1
fi
OFIEOF
    srun --overlap --nodelist=${HOSTS[0]},${HOSTS[1]} --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=none bash $LOGD/ofi-serve.sh &
    PIDS+=($!)
  elif [ "$a" = one ]; then
    comm one 1 4
    GP=0,1,2,3
    srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 --cpu-bind=none bash -c "export VLLM_HOST_IP=\$($ipof); export SINGULARITYENV_VLLM_HOST_IP=\$VLLM_HOST_IP ROCR_VISIBLE_DEVICES=$GP SINGULARITYENV_ROCR_VISIBLE_DEVICES=$GP; exec singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL $FLAGS --limit-mm-per-prompt '$LIM' --host 127.0.0.1 --port $p" > $LOGD/serve-one.log 2>&1 &
    PIDS+=($!)
  else
    comm two 2 2
    local HEAD; HEAD=$(srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 bash -c "$ipof")
    GP=0,1
    srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 --cpu-bind=none bash -c "export VLLM_HOST_IP=\$($ipof); export SINGULARITYENV_VLLM_HOST_IP=\$VLLM_HOST_IP ROCR_VISIBLE_DEVICES=$GP SINGULARITYENV_ROCR_VISIBLE_DEVICES=$GP; exec singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL $FLAGS --nnodes 2 --node-rank 0 --master-addr $HEAD --master-port 29501 --limit-mm-per-prompt '$LIM' --host 127.0.0.1 --port $p" > $LOGD/serve-two-r0.log 2>&1 &
    PIDS+=($!)
    srun --overlap --nodelist=${HOSTS[1]} --nodes=1 --ntasks=1 --cpu-bind=none bash -c "export VLLM_HOST_IP=\$($ipof); export SINGULARITYENV_VLLM_HOST_IP=\$VLLM_HOST_IP ROCR_VISIBLE_DEVICES=$GP SINGULARITYENV_ROCR_VISIBLE_DEVICES=$GP; exec singularity exec -B /pfs,/scratch,/flash $SIF vllm serve $MODEL $FLAGS --nnodes 2 --node-rank 1 --master-addr $HEAD --master-port 29501 --headless --limit-mm-per-prompt '$LIM'" > $LOGD/serve-two-r1.log 2>&1 &
    PIDS+=($!)
  fi
  for ((w=0; w<UP_LIMIT; w+=10)); do
    on0 singularity exec $SIF python3 -c "import urllib.request as u; u.urlopen('http://127.0.0.1:$p/v1/models', timeout=5)" >/dev/null 2>&1 && break
    for q in "${PIDS[@]}"; do kill -0 "$q" 2>/dev/null || dead=$q; done
    [ -n "$dead" ] && break
    sleep 10
  done
  if [ -n "$dead" ] || [ $w -ge $UP_LIMIT ]; then
    echo "## $a server NOT UP (dead=$dead waited=$w s) $(date +%T)"
    stop; return 4
  fi
  echo "## $a server up after $(( $(date +%s) - t0 )) s $(date +%T)"
  grep -hE "GPU KV cache size|Maximum concurrency" $LOGD/serve-$a*.log | sed -E 's/^.*INFO[^]]*\] *//' | head -3
  on0 timeout $CHECK_LIMIT singularity exec -B /pfs,/scratch,/flash $SIF python3 $SERVE_CHECK $p mn --needle 2>&1 | cut -c1-300
  echo "## $a check rc=${PIPESTATUS[0]}"
  on0 timeout $CLIENT_LIMIT singularity exec -B /pfs,/scratch,/flash $SIF python3 $SC/vllm_mn_client.py $p mn $ROUNDS 2>&1 | sed "s/^{/{\"arm\": \"$a\", /"
  echo "## $a client rc=${PIPESTATUS[0]} $(date +%T)"
  echo "## $a serve transport:"; grep -hE 'Using network|Selected Provider|Using aws-ofi' $LOGD/serve-$a*.log | sed -E 's/^.*NCCL (INFO|WARN) *//' | sort | uniq -c
  stop
  unset SINGULARITYENV_LD_LIBRARY_PATH SINGULARITYENV_FI_MR_CACHE_MONITOR SINGULARITYENV_FI_CXI_DEFAULT_CQ_SIZE
}
echo "job $SLURM_JOB_ID arms [$ARMS] start $(date -Is)"
for a in $ARMS; do arm $a; echo "## $a rc=$?"; done
echo "end $(date -Is)"
