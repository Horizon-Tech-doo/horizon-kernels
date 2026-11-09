#!/bin/bash
#SBATCH --job-name=ofi-comm
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=80G
#SBATCH --time=00:14:00
#SBATCH --output=%x-%j.out
# Track 5: RCCL inside the vLLM image across two nodes, 2 GCDs each, with and without LUMI's OFI plugin. Each arm is one step with a short limit.
#   sock: the image as shipped. asis: LUMI's plugin named libnccl-net.so, only missing libraries shimmed. adapter: hand v6 adapter over LUMI's v5 plugin.
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
SC=$TL/scaling; OFI=${OFI_DIR:-$SC/ofi}   # OFI_DIR: adapter/, asis/, shim/, aws-ofi-rccl/, libfabric/, shs-libcxi/ as described in the release README (Scaling)
SIF=${VLLM_SIF:-/flash/$PROJ/vllm-check/images/vllm-rocm-v0.30.0.sif}
ARMS=${ARMS:-sock asis adapter sock adapter}
LOGD=$TL/logs/ofi-comm-$SLURM_JOB_ID; mkdir -p $LOGD
HOSTS=($(scontrol show hostnames $SLURM_JOB_NODELIST))
export SINGULARITYENV_PYTHONNOUSERSITE=1 SINGULARITYENV_NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3 SINGULARITYENV_NCCL_NET_GDR_LEVEL=PHB SINGULARITYENV_NCCL_DEBUG=INFO SINGULARITYENV_NCCL_DEBUG_SUBSYS=INIT,NET
export MASTER_PORT=29610
L=$OFI/shim:$OFI/aws-ofi-rccl:$OFI/libfabric/lib:$OFI/shs-libcxi/lib:/opt/rocm-7.2.3/lib
echo "job $SLURM_JOB_ID arms [$ARMS] start $(date -Is)"
i=0
for a in $ARMS; do
  i=$((i+1)); unset SINGULARITYENV_LD_LIBRARY_PATH SINGULARITYENV_FI_MR_CACHE_MONITOR SINGULARITYENV_FI_CXI_DEFAULT_CQ_SIZE
  case $a in
    asis) export SINGULARITYENV_LD_LIBRARY_PATH=$OFI/asis:$L ;;
    adapter) export SINGULARITYENV_LD_LIBRARY_PATH=$OFI/adapter:$L ;;
  esac
  if [ $a != sock ]; then export SINGULARITYENV_FI_MR_CACHE_MONITOR=userfaultfd SINGULARITYENV_FI_CXI_DEFAULT_CQ_SIZE=131072; fi
  export MASTER_PORT=$((MASTER_PORT+1))
  export MASTER_ADDR=$(srun --overlap --nodelist=${HOSTS[0]} --nodes=1 --ntasks=1 bash -c 'ip -4 -o addr show hsn0 | sed -E "s/.*inet ([0-9.]+).*/\1/"')
  echo "## $i $a start $(date +%T)"
  timeout 170 srun --overlap --nodes=2 --ntasks=4 --ntasks-per-node=2 --cpus-per-task=7 --cpu-bind=none --nodelist=${HOSTS[0]},${HOSTS[1]} \
    bash -c 'export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID SINGULARITYENV_ROCR_VISIBLE_DEVICES=$SLURM_LOCALID; [ "$SLURM_PROCID" = 0 ] && env | grep -E "^SLINGSHOT_(VNIS|TCS)" | sed "s/^/ENV /"; exec singularity exec -B /pfs,/scratch,/flash '$SIF' python3 '$SC'/vllm_mn_comm.py' > $LOGD/comm-$i-$a.log 2>&1
  echo "## $i $a rc=$? end $(date +%T)"
  grep -oE '\{"(op|check).*\}' $LOGD/comm-$i-$a.log | sed "s/^{/{\"arm\": \"$a\", \"seq\": $i, /"
  grep -E 'NET/|Using network|symbol|cannot open|ENV SLINGSHOT|libfabric|not found' $LOGD/comm-$i-$a.log | sed -E 's/^[^ ]+ +[^ ]+ +//' | cut -c1-170 | sort | uniq -c | sort -rn | head -14
done
echo "end $(date -Is)"
