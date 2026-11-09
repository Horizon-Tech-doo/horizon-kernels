#!/bin/bash
# Inside the container, cwd exp/N4.
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4 || exit 1
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
$HC -c -x hip probe.cu -o probe.co 2>&1 | grep -E "error"
hipcc -O2 -std=c++17 hbn.cpp -o hbn -lhipblaslt 2>&1 | grep -E "error|cannot|undefined" | head
ls -la hbn probe.co
