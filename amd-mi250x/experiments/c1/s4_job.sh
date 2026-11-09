#!/bin/bash
# Patch tests on GPU, fork build. Also the same bug tests on the wheel to show they fail without the patches.
R=/scratch/project_XXXXXXXXX/tilelang; C=$R/exp/C1; cd $C; J=$SLURM_JOB_ID; O=raw/s4_$J; mkdir -p $O
export ROCM_PATH=$R/hipcc-wrap
PYF="PYTHONPATH=$R/fork LD_LIBRARY_PATH=$R/fork/build/lib:$LD_LIBRARY_PATH TILELANG_CACHE_DIR=$C/cache_t"
cd $R/fork
env $PYF timeout 700 python -m pytest -q -x --no-header -p no:cacheprovider testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py testing/python/amd/test_tilelang_hip_uint8_broadcast.py testing/python/amd/test_tilelang_hip_gemm_small_tile.py > $C/$O/pytest_new.txt 2>&1; echo "new tests rc=$?"
env $PYF timeout 500 python -m pytest -q --no-header -p no:cacheprovider testing/python/amd/test_tilelang_hip_codegen.py testing/python/amd/test_tilelang_hip_gemm_block_size.py testing/python/amd/test_tilelang_hip_vector_select_codegen.py testing/python/amd/test_tilelang_test_amd.py > $C/$O/pytest_regress.txt 2>&1; echo "regress rc=$?"
cd $C
env $PYF C9_OUT=$O/c9_err_fork timeout 400 python c9_repro.py all 2>&1 | grep -E "RESULT|^tilelang" > $O/c9_fork.txt
tail -5 $O/pytest_new.txt; tail -5 $O/pytest_regress.txt; date
