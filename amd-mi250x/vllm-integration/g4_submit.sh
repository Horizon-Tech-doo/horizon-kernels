#!/bin/bash
# G4: query-window prefill attention (gqa_fa_qw.py, HLP_QW=1) with the GEMM plugin, against stock and against the GEMM plugin alone; two settings x two mirrored orders, whole nodes. bash g4_submit.sh
# Output: /scratch/project_XXXXXXXXX/tilelang/exp/G1/raw/g4-<setting>-<a|b>-<jobid>.out and .../raw/<jobid>/ (client-*.jsonl, eq-*.jsonl, ctr-*/, power.csv, serve-*.log).
L=/scratch/project_XXXXXXXXX/tilelang/exp/G1; VI=$L/vi5; cd $VI
X="PROJ=project_XXXXXXXXX,VI=$VI,E2E_USE=0:1"; S="-A project_XXXXXXXXX -p standard-g --gpus-per-node=8 --no-requeue"
sbatch $S -t 01:45:00 -J g4-def-a -o $L/raw/%x-%j.out --export=ALL,$X,ARMS='fgwv_rocm s_rocm g_rocm fgw_rocm fgw_rocm g_rocm s_rocm' g4.sbatch
sbatch $S -t 01:30:00 -J g4-def-b -o $L/raw/%x-%j.out --export=ALL,$X,ARMS='fgw_rocm g_rocm s_rocm s_rocm g_rocm fgw_rocm' g4.sbatch
sbatch $S -t 01:30:00 -J g4-one-a -o $L/raw/%x-%j.out --export=ALL,$X,MNBT=16384,ARMS='s_rocm g_rocm fgw_rocm fgw_rocm g_rocm s_rocm' g4.sbatch
sbatch $S -t 01:30:00 -J g4-one-b -o $L/raw/%x-%j.out --export=ALL,$X,MNBT=16384,ARMS='fgw_rocm g_rocm s_rocm s_rocm g_rocm fgw_rocm' g4.sbatch
