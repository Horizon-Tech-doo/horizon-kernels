#!/bin/bash
#SBATCH --job-name=p1-qw-check
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:30:00
# P1 gate for gqa_fa_qw.py on one GCD: float64 on dumped server activations and random inputs, bitwise comparison with the square kernel, timings with an identical-code control. P1_REV=1 for the second job.
TL=/scratch/project_XXXXXXXXX/tilelang; E=$TL/exp/P1; VI=$E/vi5; SIF=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif; O=$E/raw/$SLURM_JOB_ID; mkdir -p $O
export HLP_SCRIPTS=$VI/hld-src TILELANG_CACHE_DIR=$E/cache-tl HLP_LADDER=1024:16384:0.06 HLP_QW=1
{ echo "## job $SLURM_JOB_ID rev=${P1_REV:-0}"; md5sum $VI/hld-src/gqa_fa.py $VI/hld-src/gqa_fa_ps.py $VI/hld-src/gqa_fa_qw.py $VI/overlay/horizon_lumi_prefill/__init__.py $VI/p1_qw_check.py | sed "s# .*/# #"; } | tee $O/header.txt
singularity exec -B /pfs,/scratch,/flash $SIF bash -c "export PYTHONNOUSERSITE=1 P1_REV=${P1_REV:-0} PYTHONPATH=$VI/overlay:$TL/venv/lib/python3.12/site-packages; python3 $VI/p1_qw_check.py $O/qw.jsonl $E/dump/ctx" > $O/qw.log 2>&1; echo "## rc=$? $(date +%T)"; grep -c '"kind": "cell"' $O/qw.jsonl; grep -h "Traceback\|Error" -A4 $O/qw.log | cut -c1-300 | tail -14; tail -1 $O/qw.jsonl | cut -c1-400
echo "## job end $(date +%T)"
