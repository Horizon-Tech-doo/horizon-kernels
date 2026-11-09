#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:20:00
#SBATCH --job-name=C2P_t4
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# The 3 test files that failed with the series, on clean main, then on the series. Same stack, one job. Also keeps the error text.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O
T="testing/python/amd/test_tilelang_gemm_mfma_intrinsic.py testing/python/amd/test_tilelang_gemm_mfma_preshuffle.py testing/python/amd/test_tilelang_mxfp4_gfx950.py"
for n in base main; do
FF=$TL_ROOT/fork-$n
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$FF TILELANG_CACHE_DIR=$E/cache/T4$n LD_LIBRARY_PATH=$FF/build/lib:\$LD_LIBRARY_PATH; cd $FF; python -c 'import tilelang;print(tilelang.__file__)'; python -m pytest -q --no-header -p no:cacheprovider --tb=line $T" > $O/$n.txt 2>&1
grep -E "^FAILED|^ERROR" $O/$n.txt | sed 's/ - .*//' | sort > $O/$n.fail
tail -1 $O/$n.txt
done
wc -l $O/*.fail; diff $O/base.fail $O/main.fail && echo SAME_FAILURES; date; echo DONE
