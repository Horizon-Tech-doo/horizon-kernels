#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --job-name=C2P_i2
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# vLLM image, ROCm 7.2.3, clang 22, final tree. The regression suite, same selection as C2P_r1. Fresh kernel cache.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; C=$E/r5/cache/i2_$SLURM_JOB_ID; mkdir -p $C
md5sum $F/build/lib/libtilelang.so > $O/lib_md5.txt
singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH; cd $F; python3 -I -c 'import sys; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$F\"); import pytest, tilelang; print(tilelang.__file__); sys.exit(pytest.main(sys.argv[1:]))' -q --no-header -p no:cacheprovider --tb=line --continue-on-collection-errors testing/python/amd testing/python/language/test_tilelang_language_reducer_v2.py -k 'not min_blocks and not uint8_broadcast and not small_tile and not occupancy_report and not compile_flags and not reduce_workspace and not reducer_scale and not stage_registers and not gemm_schedule'" > $O/pytest.txt 2>&1
grep -E "^(FAILED|ERROR)" $O/pytest.txt | sed 's/ - .*//' | sort > $O/fail.txt; wc -l $O/fail.txt; tail -1 $O/pytest.txt | cut -c1-200
rm -rf $C; date; echo DONE
