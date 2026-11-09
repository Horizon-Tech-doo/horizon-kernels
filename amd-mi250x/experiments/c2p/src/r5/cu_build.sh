#!/bin/bash
# Inside the vLLM image: it has gcc, which nvcc needs as host compiler; the LUMI container does not. CPU only. Series tree with CUDA, ROCm and Ascend on, as a Linux wheel is built. Out-of-tree build dir, fork-main/build untouched.
set -x
R=/scratch/project_XXXXXXXXX/tilelang; F=$R/fork-main; T=$R/exp/C2P/r5; B=$T/build-cuda
export WITH_PIP_CUDA_TOOLCHAIN=$T/tk/nvidia/cu13
export PYTHONNOUSERSITE=1 CCACHE_DISABLE=1
$WITH_PIP_CUDA_TOOLCHAIN/bin/nvcc --version | head -2
mkdir -p $B
export CC=gcc CXX=g++
gcc --version | head -1
cmake -S $F -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DTILELANG_USE_CUDA_STUBS=ON -DUSE_ROCM=ON -DTILELANG_USE_HIP_STUBS=ON -DUSE_ASCEND=ON -DTILELANG_USE_ASCEND_STUBS=ON -DUSE_METAL=OFF -DPython_EXECUTABLE=$(which python3) -DPython3_EXECUTABLE=$(which python3) 2>&1 | tail -25
grep -E '^(USE_CUDA|USE_ROCM|USE_ASCEND|TILELANG_USE_CUDA_STUBS)|TILELANG_CUDA_TOOLKIT_(AVAILABLE|SOURCE)' $B/CMakeCache.txt
grep -c 'src/cuda/codegen/codegen_cuda' $B/build.ninja
time ninja -C $B -j ${SLURM_CPUS_PER_TASK:-16} -k 0 tilelang > $T/ninja.log 2>&1; echo "ninja rc=$?"
grep -E ' error[: ]|^FAILED' $T/ninja.log | cut -c1-200 | head -40; tail -2 $T/ninja.log | cut -c1-200
ls -la $B/lib | cut -c1-120
echo BUILD_DONE
