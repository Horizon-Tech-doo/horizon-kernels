#!/bin/bash
cd /scratch/project_XXXXXXXXX/tilelang/exp/E1
J=$SLURM_JOB_ID
AMD_LOG_LEVEL=4 ./hbm 1 2>&1 >/dev/null | grep -aE "ShaderName|grid=" | grep -aE "Cijk|grid=" | sed -E "s/^.*(ShaderName)/\1/; s/^.*(grid=)/\1/" | cut -c1-200 | uniq -c > out/hbgrid_$J.txt
wc -l out/hbgrid_$J.txt
