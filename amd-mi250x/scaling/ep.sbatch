#!/bin/bash
#SBATCH --job-name=ep-moe
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=00:12:00
#SBATCH --output=%x-%j.out
# Track 5: expert-parallel MoE. One allocation runs the worlds in EP_WORLDS in order, so they compare within one job:
#   EP_WORLDS="1 8 16 32" sbatch --nodes=4 --gpus-per-node=8 --ntasks-per-node=8 ep.sbatch
# Fill the kernel cache first, on one GCD:
#   EP_PRECOMPILE=1,8,16,32 sbatch -p small-g --nodes=1 --gpus-per-node=1 --ntasks-per-node=1 --mem=60G --time=00:20:00 ep.sbatch
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
PROJ=${PROJ:?set PROJ to your LUMI project}; TL=${TL_ROOT:-/scratch/$PROJ/tilelang}
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
export EP_TOKENS=${EP_TOKENS:-512,2048} EP_MOE_MD5=${EP_MOE_MD5:-9ad0c3452cd666efe2782f7ac3abe0c4} EP_DIRECT=${EP_DIRECT:-1}
export EP_CACHE=${EP_CACHE:-$TL/scaling/cache}
mkdir -p $EP_CACHE
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
[ "${SLURM_NTASKS_PER_NODE:-8}" = 8 ] || CPU_BIND=none     # the 8-GCD masks don't fit a smaller allocation
# MASTER_ADDR from the step's own node list: rank 0 is on its first node, and a step smaller than the job need not start on the job's first node.
RUN='export MASTER_ADDR=$(scontrol show hostnames $SLURM_STEP_NODELIST | head -n1); [ "$SLURM_PROCID" = 0 ] && echo "## step nodes $SLURM_STEP_NUM_NODES tasks $SLURM_STEP_NUM_TASKS rank0_is_master $([ "$(hostname)" = "$MASTER_ADDR" ] && echo yes || echo no)"; export ROCR_VISIBLE_DEVICES=${EP_DEV:-$SLURM_LOCALID}; singularity exec -B /pfs,/scratch,/flash '"$SIF"' bash -c "\$WITH_CONDA; source '"$TL"'/venv/bin/activate; export ROCM_PATH='"$TL"'/hipcc-wrap PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR='"$EP_CACHE"' TRITON_CACHE_DIR='"$EP_CACHE"'/triton; python '"$TL"'/scaling/ep_moe.py"'
if [ -n "${EP_PRECOMPILE:-}" ]; then
  srun --ntasks=1 --cpu-bind=none bash -c "$RUN"
  exit
fi
PORT=29512
for W in ${EP_WORLDS:-${SLURM_NTASKS:-1}}; do
  N=$(( (W + 7) / 8 )); PER=$(( W < 8 ? W : 8 ))
  export MASTER_PORT=$PORT; PORT=$((PORT + 1))
  echo "## world $W nodes $N start $(date +%T)"
  # One task gets cores 1-7 from Slurm, which are the cores next to GCD 4 in the mask list, so W = 1 runs on that GCD. The timeout stops a hang early.
  if [ "$W" = 1 ]; then BIND=none; export EP_DEV=4; else BIND=$CPU_BIND; unset EP_DEV; fi
  timeout ${EP_STEP_LIMIT:-170} srun --nodes=$N --ntasks=$W --ntasks-per-node=$PER --cpu-bind=$BIND bash -c "$RUN"
  echo "## world $W rc $? end $(date +%T)"
done
