#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4; mkdir -p r5
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(CUDA_VISIBLE_DEVICES= python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
CUDA_VISIBLE_DEVICES= python gen_r5d.py 2>&1 | tail -3
ls r5/r5d_*.cu | sed 's/\.cu$//' | xargs -P 4 -I{} sh -c "$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I. -c -x hip {}.cu -o {}.co 2>&1 | grep ' error' | head -2"
echo co $(ls r5/r5d_*.co | wc -l) cu $(ls r5/r5d_*.cu | wc -l)
