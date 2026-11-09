#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --job-name=C2P_k2
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# LUMI stack, final tree. The 4 kpool test files that never collect, because the container's site-packages has its own `examples`. Same tests, with the repo's examples put first on that package's path.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O; C=$E/r5/cache/k1_$SLURM_JOB_ID; mkdir -p $C
cat > $O/run.py <<P
import sys, examples, pytest
examples.__path__.insert(0, "$F/examples")
sys.exit(pytest.main(["-q", "--no-header", "-p", "no:cacheprovider", "--tb=short", "--continue-on-collection-errors", "--maxfail=3"] + ["testing/python/amd/test_tilelang_hip_glm53_kpool_%s.py" % n for n in ("compress", "decode_tail", "fp8_mqa_logits", "topk_transform")]))
P
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$F:$S TILELANG_CACHE_DIR=$C LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH; cd $F; python $O/run.py" > $O/pytest.txt 2>&1
grep -E "^(FAILED|ERROR)" $O/pytest.txt | sed 's/ - .*//' | sort > $O/fail.txt; wc -l $O/fail.txt; tail -1 $O/pytest.txt | cut -c1-200
rm -rf $C; date; echo DONE
