#!/bin/bash
# Inside the image, CPU only. Clean upstream 6e8f599e from git archive, no worktree, so fork-main is untouched. Removed after use.
set -x
R=/scratch/project_XXXXXXXXX/tilelang; M=$R/fork-main; T=$R/exp/C2P/r5; F=$T/base; B=$F/build
rm -rf $F; mkdir -p $F; git -C $M archive 6e8f599e2fa41f05bb812edd8d06985b04f4f18b | tar xf - -C $F
rm -rf $F/3rdparty; ln -sfn $M/3rdparty $F/3rdparty
export CC=gcc CXX=g++ CCACHE_DISABLE=1 PYTHONNOUSERSITE=1; mkdir -p $B
cmake -S $F -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DTILELANG_USE_CUDA_STUBS=OFF -DUSE_ROCM=ON -DTILELANG_USE_HIP_STUBS=ON -DUSE_ASCEND=OFF -DUSE_METAL=OFF -DPython_EXECUTABLE=$(which python3) -DPython3_EXECUTABLE=$(which python3) 2>&1 | tail -3
ninja -C $B -j ${SLURM_CPUS_PER_TASK:-16} -k 0 tilelang > $T/ninja_base.log 2>&1; echo "ninja rc=$?"; tail -1 $T/ninja_base.log | cut -c1-120
ln -sf $R/venv/lib/python3.12/site-packages/tilelang/lib/tilelang_cython_wrapper.abi3.so $B/lib/
echo BUILD_DONE
