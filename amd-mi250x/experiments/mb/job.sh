#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/analysis/mb
rocm-smi --showclocks 2>/dev/null | grep -E "sclk|GPU\[0\]" | head -4
timeout 600 ./mb
rc=$?
echo rc=$rc
rocm-smi --showclocks 2>/dev/null | grep -E "sclk" | head -2
