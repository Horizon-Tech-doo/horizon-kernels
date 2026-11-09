#!/bin/bash
# Inside the LUMI PyTorch container, using the container's clang++ (libstdc++ 12 headers). CPU-only.
set -x
F=/scratch/project_XXXXXXXXX/tilelang/fork; B=$F/build
cd $F; mkdir -p $B
export CC=/opt/rocm-6.2.4/llvm/bin/clang CXX=/opt/rocm-6.2.4/llvm/bin/clang++
python -c "import z3;print(z3.__file__)"
[ -f $B/build.ninja ] || cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF -DUSE_ROCM=ON -DUSE_METAL=OFF -DUSE_ASCEND=OFF -DTILELANG_USE_HIP_STUBS=ON -DTILELANG_USE_CUDA_STUBS=OFF -DPython_EXECUTABLE=$(which python) -DPython3_EXECUTABLE=$(which python) 2>&1 | tail -40
time ninja -C $B -j ${SLURM_CPUS_PER_TASK:-16} -k 3 tilelang 2>&1 | tail -30
ls -la $B/lib
du -sh $B
