#!/bin/bash
#SBATCH --job-name=rccl-scale
#SBATCH --partition=dev-g
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.out
# Track 5: RCCL all-reduce / all-to-all on N nodes x 8 GCDs. Submit: sbatch --nodes=N comm.sbatch
PROJ=${PROJ:?set PROJ to your LUMI project}; export TL_ROOT=${TL_ROOT:-/scratch/$PROJ/tilelang}
SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29511
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3      # Slingshot interfaces (LUMI docs)
export NCCL_NET_GDR_LEVEL=PHB
export MIOPEN_USER_DB_PATH=/tmp/miopen-$SLURM_JOB_ID
# one GCD per task, CPU cores close to that GCD (LUMI-G CPU/GPU affinity)
CPU_BIND="mask_cpu:fe000000000000,fe00000000000000,fe0000,fe000000,fe,fe00,fe00000000,fe0000000000"
srun --cpu-bind=$CPU_BIND bash -c 'export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID; singularity exec -B /pfs,/scratch,/flash '"$SIF"' bash -c "\$WITH_CONDA; python $TL_ROOT/scaling/comm_bench.py"'
