#!/bin/bash
#SBATCH --account=project_XXXXXXXXX
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=00:19:00
#SBATCH --job-name=C2P_k3
#SBATCH --output=/scratch/project_XXXXXXXXX/tilelang/exp/C2P/out/slurm-%j.out
# Image, clang 22. The 4 kpool test files on the series, then on clean main, same job, 8 minutes each, verbose so a stall names its test.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
O=$E/out/$SLURM_JOB_ID; mkdir -p $O
T="testing/python/amd/test_tilelang_hip_glm53_kpool_compress.py testing/python/amd/test_tilelang_hip_glm53_kpool_decode_tail.py testing/python/amd/test_tilelang_hip_glm53_kpool_fp8_mqa_logits.py testing/python/amd/test_tilelang_hip_glm53_kpool_topk_transform.py"
for n in series base; do
[ $n = series ] && FF=$F || FF=$E/r5/base
C=$E/r5/cache/k3_${n}_$SLURM_JOB_ID; mkdir -p $C
echo "=== $n $(date +%T)"
singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$C HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=$FF/build/lib:\$LD_LIBRARY_PATH; cd $FF; timeout 480 python3 -I -c 'import sys; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$FF\"); import pytest, tilelang; print(tilelang.__file__); sys.exit(pytest.main(sys.argv[1:]))' -v --no-header -p no:cacheprovider --tb=line --continue-on-collection-errors $T; echo rc=\$?" > $O/$n.txt 2>&1
grep -E "::.* (PASSED|FAILED|SKIPPED|ERROR)" $O/$n.txt | sed -E 's/ +\[ *[0-9]+%\]//' > $O/$n.res; grep -E "::" $O/$n.txt | tail -1 | cut -c1-200; tail -1 $O/$n.txt | cut -c1-160; wc -l < $O/$n.res
rm -rf $C
done
diff $O/series.res $O/base.res && echo SAME_RESULTS; date; echo DONE
