#!/bin/bash
# Fork tree on scratch, v0.1.15 + submodules from local clones already on scratch. No network needed; pinned commits verified.
set -e
R=/scratch/project_XXXXXXXXX/tilelang; F=$R/fork
test -d $F/.git || git clone -q --shared $R/src-v0.1.15 $F
cd $F; git checkout -q -B c1 a35f8ddf45eba16c21211ec8822d56ce5363036f
rm -rf 3rdparty/tvm; git clone -q --shared $R/src/3rdparty/tvm 3rdparty/tvm
cd 3rdparty/tvm; git checkout -q 4211874e9e4b8770fd20ad0d98ab80afd1df5028
rm -rf 3rdparty/tvm-ffi; git clone -q --shared $R/src/3rdparty/tvm/3rdparty/tvm-ffi 3rdparty/tvm-ffi
git -C 3rdparty/tvm-ffi checkout -q 3c35034fd1026011736e19a4e0e1ed0f22058c42
for s in dlpack libbacktrace; do rm -rf 3rdparty/tvm-ffi/3rdparty/$s; git clone -q --shared $R/src/3rdparty/tvm/3rdparty/tvm-ffi/3rdparty/$s 3rdparty/tvm-ffi/3rdparty/$s; done
git submodule status --recursive 2>/dev/null | head -12
cd $F; git log --oneline -1; git -C 3rdparty/tvm log --oneline -1; git -C 3rdparty/tvm-ffi log --oneline -1
