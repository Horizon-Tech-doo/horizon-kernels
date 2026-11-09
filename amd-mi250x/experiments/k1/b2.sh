#!/bin/bash
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I."
$HC -c -x hip k1_micro.cu -o micro.co 2>&1 | grep -E "error|illegal" | head; $HC -S -x hip k1_micro.cu -o micro.s 2>&1 | grep -E " error" | head -3
hipcc -O2 -std=c++17 main.cpp -o k1 2>&1 | grep -E "error" | head
ls -la micro.co k1
