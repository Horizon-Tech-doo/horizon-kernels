#!/bin/bash
# Incremental rebuild of fork-main on the login node. Inside the container, -j8.
R=/scratch/project_XXXXXXXXX/tilelang; F=$R/fork-main
ln -sf $R/venv/lib/python3.12/site-packages/tilelang/lib/tilelang_cython_wrapper.abi3.so $F/build/lib/
$R/scripts/tl_run.sh "export CC=/opt/rocm-6.2.4/llvm/bin/clang CXX=/opt/rocm-6.2.4/llvm/bin/clang++; ninja -C $F/build -j 8 tilelang 2>&1 | grep -E 'error|FAILED|Linking|no work' | head -20"
