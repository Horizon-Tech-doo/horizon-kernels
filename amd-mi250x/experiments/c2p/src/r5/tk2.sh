#!/bin/bash
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
R=$E/r5; cd $R; : > tk2.log
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate;
for p in nvidia-cuda-cudart-dev nvidia-cudart-dev nvidia-cuda-cudart nvidia-cudart nvidia-cuda-runtime nvidia-cuda-runtime-dev nvidia-cuda-driver-dev nvidia-cuda-driver nvidia-cuda-nvrtc nvidia-cuda-nvrtc-dev; do
 echo \"== \$p\"; timeout 150 python -m pip install --no-deps --no-cache-dir --upgrade --target $R/tk \$p 2>&1 | grep -E 'Successfully|ERROR' | cut -c1-160; done" >> tk2.log 2>&1
find tk -name 'cuda_runtime.h' -o -name 'libcudart*' -o -name 'cuda.h' -o -name 'nvrtc.h' | cut -c1-120 >> tk2.log
echo TK2_DONE >> tk2.log
