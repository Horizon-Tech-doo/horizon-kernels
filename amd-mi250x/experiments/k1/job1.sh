#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/K1
hostname; date -u; nvidia=$(rocm-smi --showproductname 2>/dev/null | grep -m1 "Card series"); echo "$nvidia"
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|mclk" | head -2
echo LIM; timeout 120 ./k1 lim; echo rc=$?
echo MICRO; timeout 1200 ./k1 micro; echo rc=$?
rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -1
date -u
