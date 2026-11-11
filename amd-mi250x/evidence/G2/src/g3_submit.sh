#!/bin/bash
# G3: the prefill-attention plugin on the weight-scaled kernel (gqa_fa_ps.py), with the GEMM plugin, against stock; two settings x two mirrored orders, whole nodes. bash g3_submit.sh
L=/scratch/project_XXXXXXXXX/tilelang/exp/G1; VI=$L/vi4; cd $VI
X="PROJ=project_XXXXXXXXX,VI=$VI,E2E_USE=0:1"; S="-A project_XXXXXXXXX -p standard-g --gpus-per-node=8 -t 01:15:00"
sbatch $S -J g3-one-a -o $L/raw/%x-%j.out --export=ALL,$X,MNBT=16384,ARMS='s_rocm fgp_rocm fgp_rocm s_rocm' g3.sbatch
sbatch $S -J g3-one-b -o $L/raw/%x-%j.out --export=ALL,$X,MNBT=16384,ARMS='fgp_rocm s_rocm s_rocm fgp_rocm' g3.sbatch
sbatch $S -J g3-def-a -o $L/raw/%x-%j.out --export=ALL,$X,ARMS='s_rocm cgp_rocm cgp_rocm s_rocm' g3.sbatch
sbatch $S -J g3-def-b -o $L/raw/%x-%j.out --export=ALL,$X,ARMS='cgp_rocm s_rocm s_rocm cgp_rocm' g3.sbatch
