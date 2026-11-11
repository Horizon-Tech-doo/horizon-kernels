#!/bin/bash
#SBATCH --job-name=p1-ps-check
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
# P1 gate for gqa_fa_ps.py on one GCD: accuracy on the dumped server activations against float64, and speed against the landed kernel with an identical-code control.
TL=/scratch/project_XXXXXXXXX/tilelang; E=$TL/exp/P1; VI=$E/vi4; SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif; O=$E/raw/$SLURM_JOB_ID; mkdir -p $O
export HLP_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl HLP_LADDER=1024:16384:0.06
md5sum $VI/hld-src/gqa_fa.py $VI/hld-src/gqa_fa_ps.py $VI/overlay/horizon_lumi_prefill/__init__.py $VI/p1_ps_check.py | sed "s# .*/# #" | tee $O/header.txt
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 P1_REV=${P1_REV:-0} PYTHONPATH=$VI/overlay:$TL/venv/lib/python3.12/site-packages; python3 $VI/p1_ps_check.py $O/ps.jsonl $E/dump/one $E/dump/ctx" > $O/ps.log 2>&1; echo "## rc=$? $(date +%T)"; grep -c '"kind"' $O/ps.jsonl; grep -h "Traceback\|Error" -A3 $O/ps.log | cut -c1-300 | tail -12; tail -1 $O/ps.jsonl | cut -c1-300
echo "## job end $(date +%T)"
