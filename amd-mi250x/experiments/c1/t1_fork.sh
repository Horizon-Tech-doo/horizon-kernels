#!/bin/bash
# Fork smoke test on the login node (no GPU): compile only.
R=/scratch/project_XXXXXXXXX/tilelang; cd $R/exp/C1
ln -sf $R/venv/lib/python3.12/site-packages/tilelang/lib/tilelang_cython_wrapper.abi3.so $R/fork/build/lib/
export PYTHONPATH=$R/fork TILELANG_CACHE_DIR=$PWD/cache_fork LD_LIBRARY_PATH=$R/fork/build/lib:$LD_LIBRARY_PATH
python - <<'P'
import tilelang, tvm_ffi
print(tilelang.__file__, tilelang.__version__, tvm_ffi.__file__)
import tilelang.rocm.language as T
print(T.annotate_min_blocks_per_sm)
P
