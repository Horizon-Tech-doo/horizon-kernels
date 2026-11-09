#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:25:00
#SBATCH --job-name=C2P_t1
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# Patch tests and the AMD regression suite on fork-main, LUMI stack. New tests first.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O
git -C $F rev-parse HEAD > $O/fork_head.txt; git -C $F status --short > $O/fork_status.txt
FM_ENV="TILELANG_CACHE_DIR=$E/cache/T" FM_L "-m pytest --version" 2>&1 | tail -1
cat > $O/run.sh <<'P'
cd $F
N="testing/python/amd/test_tilelang_hip_min_blocks_per_sm.py testing/python/amd/test_tilelang_hip_uint8_broadcast.py testing/python/amd/test_tilelang_hip_gemm_small_tile.py testing/python/amd/test_tilelang_hip_occupancy_report.py testing/python/amd/test_tilelang_hip_compile_flags.py testing/python/amd/test_tilelang_hip_reduce_workspace.py testing/python/amd/test_tilelang_hip_reducer_scale.py testing/python/amd/test_tilelang_hip_stage_registers.py"
python -m pytest -q --no-header -p no:cacheprovider $N 2>&1 | tail -40
python -m pytest -q --no-header -p no:cacheprovider --timeout=120 testing/python/amd testing/python/language/test_tilelang_language_reducer_v2.py -k "not min_blocks and not uint8_broadcast and not small_tile and not occupancy_report and not compile_flags and not reduce_workspace and not reducer_scale and not stage_registers" 2>&1 | tail -40
P
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$F:$S TILELANG_CACHE_DIR=$E/cache/T LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH F=$F; bash $O/run.sh" > $O/pytest.txt 2>&1
tail -30 $O/pytest.txt; date; echo DONE
