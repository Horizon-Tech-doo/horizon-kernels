#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:50:00
#SBATCH --job-name=C2P_r1
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# LUMI stack, clang 18, final tree. New tests first, then the regression suite. Fresh kernel cache.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; C=$E/r5/cache/r1_$SLURM_JOB_ID; mkdir -p $C
git -C $F rev-parse HEAD > $O/fork_head.txt; git -C $F status --short > $O/fork_status.txt; md5sum $F/build/lib/libtilelang.so > $O/lib_md5.txt
cat > $O/run.sh <<'P'
cd $F
python -m pytest -q --no-header -p no:cacheprovider testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py testing/python/amd/test_tilelang_hip_uint8_broadcast.py testing/python/amd/test_tilelang_hip_gemm_small_tile.py testing/python/amd/test_tilelang_hip_occupancy_report.py testing/python/amd/test_tilelang_hip_compile_flags.py testing/python/amd/test_tilelang_hip_reduce_workspace.py testing/python/amd/test_tilelang_hip_reducer_scale.py testing/python/amd/test_tilelang_hip_stage_registers.py testing/python/amd/test_tilelang_hip_gemm_schedule.py 2>&1 | grep -v "__tb\|<unknown>\|???" | tail -40
python -m pytest -q --no-header -p no:cacheprovider --continue-on-collection-errors testing/python/amd testing/python/language/test_tilelang_language_reducer_v2.py -k "not min_blocks and not uint8_broadcast and not small_tile and not occupancy_report and not compile_flags and not reduce_workspace and not reducer_scale and not stage_registers and not gemm_schedule" 2>&1 | tail -60
P
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$F:$S TILELANG_CACHE_DIR=$C LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH F=$F; bash $O/run.sh" > $O/pytest.txt 2>&1
grep -E "^(FAILED|ERROR)" $O/pytest.txt | sed 's/ - .*//' | sort > $O/fail.txt; wc -l $O/fail.txt; tail -1 $O/pytest.txt | cut -c1-200
rm -rf $C; date; echo DONE
