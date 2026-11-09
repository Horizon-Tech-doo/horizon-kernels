#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4; mkdir -p h11
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(CUDA_VISIBLE_DEVICES= python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
CUDA_VISIBLE_DEVICES= python gen_r3.py 2>&1 | tail -1
ls h11/h11_*.cu | sed 's/\.cu$//' | xargs -P 6 -I{} sh -c "$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL -I. -c -x hip {}.cu -o {}.co 2>&1 | grep ' error' | head -2"
ls h11/h11_*.co | wc -l; ls h11/h11_*.cu | wc -l
