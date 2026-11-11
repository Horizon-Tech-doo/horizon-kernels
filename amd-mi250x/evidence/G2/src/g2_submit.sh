#!/bin/bash
# G2 submissions, run on the LUMI login node from exp/G1/vi3: bash g2_submit.sh check|servers
# check: the landed table (gemm_bf16.ROWS_FORMS) through the plugin's own path, forward and reversed order, one GCD each.
# servers: four whole-node jobs (server on the two dies of one card), two settings x two mirrored arm orders, every arm twice.
set -u
L=/scratch/project_XXXXXXXXX/tilelang/exp/G1; VI=$L/vi3; cd $VI
X="PROJ=project_XXXXXXXXX,VI=$VI,E2E_USE=0:1"
case ${1:-check} in
  check)   sbatch -J g2-landed-fwd -o $L/raw/%x-%j.out --export=ALL,VI=$VI g2_check.sbatch
           sbatch -J g2-landed-rev -o $L/raw/%x-%j.out --export=ALL,VI=$VI,G1_REV=1 g2_check.sbatch ;;
  servers) S="-A project_XXXXXXXXX -p standard-g --gpus-per-node=8 -t 02:30:00"
           sbatch $S -J g2-one-a -o $L/raw/%x-%j.out --export=ALL,$X,MNBT=16384,ARMS='s_rocm g_rocm fg_rocm s_tri fg_tri fg_tri s_tri fg_rocm g_rocm s_rocm' g2.sbatch
           sbatch $S -J g2-one-b -o $L/raw/%x-%j.out --export=ALL,$X,MNBT=16384,ARMS='fg_tri s_tri fg_rocm g_rocm s_rocm s_rocm g_rocm fg_rocm s_tri fg_tri' g2.sbatch
           sbatch $S -J g2-def-a -o $L/raw/%x-%j.out --export=ALL,$X,ARMS='s_rocm g_rocm cg_rocm s_tri g_tri g_tri s_tri cg_rocm g_rocm s_rocm' g2.sbatch
           sbatch $S -J g2-def-b -o $L/raw/%x-%j.out --export=ALL,$X,ARMS='g_tri s_tri cg_rocm g_rocm s_rocm s_rocm g_rocm cg_rocm s_tri g_tri' g2.sbatch ;;
esac
