#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/K1
date -u; rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -2
mkdir -p bins
echo MICRO2; timeout 300 ./k1 micro2; echo rc=$?
for r in 0 1 2; do
  echo GEMM $r; timeout 500 ./k1 gemm $r bins; echo rc=$?
  echo HB $r; timeout 300 python hb.py bins $r 2>&1 | grep -v Warning; echo rc=$?
done
rm -rf bins
echo BADVA; timeout 120 ./k1 badva; echo rc=$?
rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
date -u
