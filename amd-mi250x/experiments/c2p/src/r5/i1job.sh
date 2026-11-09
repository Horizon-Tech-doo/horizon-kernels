#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:15:00
#SBATCH --job-name=C2P_i1
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# vLLM image, ROCm 7.2.3, clang 22, final tree. The new tests of the series. Fresh kernel cache.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; C=$E/r5/cache/i1_$SLURM_JOB_ID; mkdir -p $C
md5sum $F/build/lib/libtilelang.so > $O/lib_md5.txt
singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH; cd $F; /opt/rocm/llvm/bin/clang --version | head -1; python3 -I -c 'import sys; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$F\"); import pytest, tilelang; print(tilelang.__file__); sys.exit(pytest.main(sys.argv[1:]))' -q --no-header -p no:cacheprovider --tb=short testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py testing/python/amd/test_tilelang_hip_uint8_broadcast.py testing/python/amd/test_tilelang_hip_gemm_small_tile.py testing/python/amd/test_tilelang_hip_occupancy_report.py testing/python/amd/test_tilelang_hip_compile_flags.py testing/python/amd/test_tilelang_hip_reduce_workspace.py testing/python/amd/test_tilelang_hip_reducer_scale.py testing/python/amd/test_tilelang_hip_stage_registers.py testing/python/amd/test_tilelang_hip_gemm_schedule.py" > $O/pytest.txt 2>&1
grep -E "^(FAILED|ERROR)" $O/pytest.txt | sed 's/ - .*//' | sort > $O/fail.txt; wc -l $O/fail.txt; tail -1 $O/pytest.txt | cut -c1-200
rm -rf $C; date; echo DONE
