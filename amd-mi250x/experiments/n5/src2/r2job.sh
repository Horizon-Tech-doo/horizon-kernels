#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=n5r2
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/N5/out/slurm-%j.out
# 3 rounds in one job; process order rotates per round, arm order rotates by tag. All image stack.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
( while true; do echo "$(date +%T) $(rocm-smi --showpower --showclocks 2>/dev/null | grep -E "sclk|Power" | tr -s " " | cut -c1-60 | tr "\n" ";")"; sleep 15; done ) > clocks.log 2>&1 &
CLK=$!; trap "kill $CLK 2>/dev/null" EXIT
A="n5_r2.py --tag @R --check --ck --rep 50"
B="n5_r2.py --tag @R --check --ck --rep 50 --re mha --inputs big,ar"
C="n5_run.py --re r2m --tag @R --inputs randn0,big,ar --check --ck --rep 50"
P=("r2:$A" "r2:$B" "OYBD:$C")
date
for r in 1 2 3; do
  for i in 0 1 2; do
    e="${P[$(( (i + r - 1) % 3 ))]}"; fl=${e%%:*}; ar=${e#*:}; ar=${ar//@R/r$r}
    echo "=== r$r $fl $(date +%T) $ar"
    timeout 240 bash -c "source $S/n5env.sh; N5_ENV='N5_R2_ASM=$E/r2asm'; TL_I '$fl' \"$ar\"" > tmp.out 2>> err.log; echo "rc=$?"
    grep '^{' tmp.out >> run.jsonl
  done
done
rm -f tmp.out; date; echo DONE
