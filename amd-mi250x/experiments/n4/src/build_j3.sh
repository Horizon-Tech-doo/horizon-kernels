#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/N4 || exit 1
R=/scratch/project_XXXXXXXXX/tilelang
TPL=$(python -c "import tilelang.env as e; print(e.TILELANG_TEMPLATE_PATH)" 2>/dev/null)
HC="$R/hipcc-wrap/bin/hipcc -O3 --offload-arch=gfx90a --genco -std=c++17 -I$TPL"
python gen_j3.py || exit 1
for f in sw/*.cu; do b=${f%.cu}; ( $HC -c -x hip $f -o $b.co 2>&1 | grep -E " error"|head -2 ) & while [ $(jobs -r | wc -l) -ge 8 ]; do sleep 0.2; done; done; wait
ls sw/*.co | wc -l
