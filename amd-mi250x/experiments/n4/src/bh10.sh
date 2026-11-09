#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4; mkdir -p h10
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(CUDA_VISIBLE_DEVICES= python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
CUDA_VISIBLE_DEVICES= python gen10.py 2>&1 | tail -1
ls h10/h10_*.cu | sed 's/\.cu$//' | xargs -P 6 -I{} sh -c "$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I. -c -x hip {}.cu -o {}.co 2>&1 | grep ' error' | head -2"
ls h10/h10_*.co | wc -l; ls h10/h10_*.cu | wc -l
