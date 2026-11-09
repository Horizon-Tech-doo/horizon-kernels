#!/bin/bash
# sbatch job.sh <plan file> <rounds>. Plan line: <L|I> <flavour or -> <n5_run args with @R for the round tag>. Passes run in plan order; the order rotates per round.
set -uo pipefail
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; cd $O
plan=$1; rounds=$2; cp "$plan" plan.txt
N5_ENV=""
( while true; do echo "$(date +%T) $(rocm-smi --showpower --showclocks 2>/dev/null | grep -E "sclk|Power" | tr -s " " | cut -c1-60 | tr "\n" ";")"; sleep 15; done ) > clocks.log 2>&1 &
CLK=$!
trap "kill $CLK 2>/dev/null" EXIT
mapfile -t LINES < <(grep -v '^#' "$plan" | grep .)
n=${#LINES[@]}
date
for r in $(seq 1 $rounds); do
  for i in $(seq 0 $((n-1))); do
    ln="${LINES[$(( (i + (r-1)) % n ))]}"
    st=$(echo "$ln" | cut -d' ' -f1); fl=$(echo "$ln" | cut -d' ' -f2); [ "$fl" = "-" ] && fl=""
    ar=$(echo "$ln" | cut -d' ' -f3- | sed "s/@R/r$r/g")
    echo "=== r$r $st[$fl] $(date +%T) $ar"
    timeout 900 bash -c "source $S/n5env.sh; N5_ENV=''; TL_$st '$fl' \"n5_run.py $ar\"" >> run.jsonl.tmp 2>> err_$st$fl.log; echo "rc=$?"
    grep '^{' run.jsonl.tmp >> run.jsonl; : > run.jsonl.tmp
  done
done
rm -f run.jsonl.tmp
date; echo DONE
