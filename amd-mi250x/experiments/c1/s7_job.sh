#!/bin/bash
# Final tree: all new tests on GPU.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; mkdir -p $C/raw; cd $R/fork
PYTHONPATH=$R/fork LD_LIBRARY_PATH=$R/fork/build/lib:$LD_LIBRARY_PATH TILELANG_CACHE_DIR=$C/cache_t ROCM_PATH=$R/hipcc-wrap timeout 500 python -m pytest -q --no-header -p no:cacheprovider -rs testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py testing/python/amd/test_tilelang_hip_uint8_broadcast.py testing/python/amd/test_tilelang_hip_gemm_small_tile.py testing/python/amd/test_tilelang_hip_occupancy_report.py > $C/raw/s7_$SLURM_JOB_ID.txt 2>&1; echo rc=$? >> $C/raw/s7_$SLURM_JOB_ID.txt; rm -rf $C/cache_t
