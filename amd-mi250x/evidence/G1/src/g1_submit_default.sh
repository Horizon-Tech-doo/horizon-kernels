#!/bin/bash
# G1 at vLLM's default chunking (1568- and 784-row steps), plugin version 2 with the per-shape tiles of the sweep (job 23624491). Run on the LUMI login node: bash g1_submit_default.sh check|node
set -u
L=/scratch/project_XXXXXXXXX/tilelang/exp/G1; VI=$L/vi2
F='17408x5120:1568:224x224:xs4;5120x8704:1568:stock;8192x5120:1568:160x256:xs4;7168x5120:1568:224x160:xs4;5120x3072:1568:160x256:xs4;17408x5120:784:stock;5120x8704:784:160x256:xs4;8192x5120:784:224x160:plw;7168x5120:784:128x128:xs4;5120x3072:784:stock'
X="PROJ=project_XXXXXXXXX,VI=$VI,HLG_MS=784:1568,HLG_MIN=512,HLG_FORMS=$F"
cd $VI
case ${1:-check} in
  check) sbatch -J g1-check-d -o $L/raw/%x-%j.out --export=ALL,$X g1_check.sbatch ;;
  node)  sbatch -A project_XXXXXXXXX -p standard-g --gpus-per-node=8 -t 01:50:00 -J g1d-node-a -o $L/raw/%x-%j.out --export=ALL,$X,E2E_USE=0:1,ARMS='gv_rocm s_rocm g_rocm fg_rocm fg_rocm g_rocm s_rocm' g1.sbatch
         sbatch -A project_XXXXXXXXX -p standard-g --gpus-per-node=8 -t 01:40:00 -J g1d-node-b -o $L/raw/%x-%j.out --export=ALL,$X,E2E_USE=0:1,ARMS='fg_rocm g_rocm s_rocm s_rocm g_rocm fg_rocm' g1.sbatch ;;
esac
