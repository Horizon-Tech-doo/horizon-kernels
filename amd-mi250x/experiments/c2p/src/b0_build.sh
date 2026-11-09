#!/bin/bash
# Clean upstream main 6e8f599e beside the series tree, to separate pre-existing test failures. Inside the container. CPU only. Removed after use.
set -x
R=/scratch/project_XXXXXXXXX/tilelang; M=$R/fork-main; F=$R/fork-base; B=$F/build
[ -d $F ] || git -C $M worktree add --detach $F 6e8f599e2fa41f05bb812edd8d06985b04f4f18b
rm -rf $F/3rdparty; ln -sfn $M/3rdparty $F/3rdparty
cd $F; mkdir -p $B
export CC=/opt/rocm-6.2.4/llvm/bin/clang CXX=/opt/rocm-6.2.4/llvm/bin/clang++
[ -f $B/build.ninja ] || cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_ROCM=ON -DUSE_METAL=OFF -DUSE_ASCEND=OFF -DTILELANG_USE_HIP_STUBS=ON -DTILELANG_USE_CUDA_STUBS=OFF -DPython_EXECUTABLE=$(which python) -DPython3_EXECUTABLE=$(which python) 2>&1 | tail -3
time ninja -C $B -j ${SLURM_CPUS_PER_TASK:-16} -k 3 tilelang 2>&1 | tail -3
ln -sf $R/venv/lib/python3.12/site-packages/tilelang/lib/tilelang_cython_wrapper.abi3.so $B/lib/
ls -la $B/lib | head -5
