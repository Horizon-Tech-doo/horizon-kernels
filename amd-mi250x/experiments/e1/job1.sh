#!/bin/bash
# step 1+2 micro, runs inside the container on the compute node
cd /scratch/project_XXXXXXXXX/tilelang/exp/E1
mkdir -p traces out
echo "== node $(hostname) job $SLURM_JOB_ID"; rocm-smi --showproductname 2>/dev/null | grep -E "Card Series" | head -1
rocminfo 2>/dev/null | grep -E "Compute Unit:|Max Waves Per CU|Wavefront Size|Max Clock Freq|Shader Engines|Shader Arrs|SIMDs per CU|Name: +gfx90a" | sort | uniq -c
rocm-smi --showclocks --showpower 2>/dev/null | grep -E "sclk|Power|power" | head
for r in 0 1 2 3 4; do
  ./mmain scan $r
done
for r in 0 1 2; do ./mmain trace $r; done
( for i in $(seq 1 30); do date +%T; rocm-smi --showclocks --showpower --showtemp --showperflevel 2>/dev/null | grep -E "sclk|Average Graphics|temperature \(edge|junction|mem\)|Performance Level"; sleep 2; done ) > out/smi_poll_$SLURM_JOB_ID.txt &
PP=$!
./mmain sustain 0 25
kill $PP 2>/dev/null
echo "== end"; rocm-smi --showclocks 2>/dev/null | grep sclk | head -1
