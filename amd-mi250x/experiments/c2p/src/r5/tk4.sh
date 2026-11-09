#!/bin/bash
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
R=$E/r5; cd $R; rm -rf tk build-cuda; : > tk4.log
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; timeout 280 python -m pip install --no-deps --no-cache-dir --target $R/tk nvidia-cuda-nvcc nvidia-cuda-cccl nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cuda-crt nvidia-nvvm 2>&1 | grep -E 'Successfully|ERROR' | cut -c1-220" >> tk4.log 2>&1
ls -d tk/nvidia/cu13/include/crt tk/nvidia/cu13/nvvm/bin >> tk4.log 2>&1
printf 'int main(){return 0;}\n' > t.cu
singularity exec -B "$TL_BIND" "$IMG" bash -c "$R/tk/nvidia/cu13/bin/nvcc -c $R/t.cu -o $R/t.o; echo nvcc_rc=\$?" >> tk4.log 2>&1
rm -f t.cu t.o; echo TK4_DONE >> tk4.log
