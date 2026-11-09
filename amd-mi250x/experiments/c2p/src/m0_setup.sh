#!/bin/bash
# Fresh clone of upstream main on scratch for the rebase. Submodules at main's pins. Login node, no build.
set -e
R=/scratch/project_XXXXXXXXX/tilelang; M=$R/fork-main
mkdir -p $R/exp/C2P
test -d $M/.git || git clone -q --filter=blob:none https://github.com/tile-ai/tilelang.git $M
cd $M; git fetch -q origin main; git checkout -q -B c2p-base 6e8f599e2fa41f05bb812edd8d06985b04f4f18b
git submodule update --init --recursive 2>&1 | tail -5 || true
git log --oneline -1; git submodule status --recursive | head
git log --oneline a35f8ddf..HEAD | wc -l
git diff --stat a35f8ddf HEAD -- src/rocm tilelang/rocm tilelang/contrib/hipcc.py tilelang/contrib/hip_resource_info.py src/op/builtin.h | tail -40
