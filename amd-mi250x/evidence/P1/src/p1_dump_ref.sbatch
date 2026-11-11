#!/bin/bash
#SBATCH --job-name=p1-dumpref
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
# float64 reference for the tensors the prefill plugin dumped in the server's verify steps (HLP_DUMP): our output and vLLM's kernel's output against the truth.
E=/scratch/project_XXXXXXXXX/tilelang/exp/P1; SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif; O=$E/raw/$SLURM_JOB_ID; mkdir -p $O
md5sum $E/vi3/p1_dump_ref.py | sed "s# .*/# #"; ls -la $E/dump/one $E/dump/ctx | cut -c1-120
for d in one ctx; do singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1; python3 $E/vi3/p1_dump_ref.py $O/dumpref-$d.jsonl $E/dump/$d" 2>&1 | grep -v "^W1\|agent.cpp" | cut -c1-600; done
echo "## job end $(date +%T)"
