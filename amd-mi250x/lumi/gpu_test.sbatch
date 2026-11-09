#!/bin/bash
#SBATCH --job-name=tl-gfx90a
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
# One GCD (= 1 GPU-h per hour). GEMM bench, AMD flash-attn fwd example, AMD unit tests, GDN compile test.
set -uo pipefail
PROJ=${PROJ:?set PROJ to your LUMI project}; HERE=${TL_SCRIPTS:-${TL_ROOT:-/scratch/$PROJ/tilelang}/scripts}   # flat scripts dir, see lumi/install.sh in the release
source $HERE/env.sh
R=$HERE/tl_run.sh
RUN=$TL_ROOT/runs/$SLURM_JOB_ID; mkdir -p $RUN; cd $RUN
step() { echo; echo "===== $1 ($(date +%T))"; }
step "gpu"; rocm-smi --showproductname 2>/dev/null | grep -E "GPU\[0\]" | head -4
step "target"; $R "python -c 'import tilelang; from tilelang.backend.target import determine_target; print(determine_target(return_object=True))'" 2>&1 | tail -3
step "reduce"; timeout 10m $R "cd $RUN && python $HERE/test_reduce.py" > reduce.log 2>&1; echo "exit $?"; grep -E "^\{|Error" reduce.log | tail -8
if [ "${SKIP_GEMM:-0}" = 0 ]; then step "gemm"; timeout 35m $R "cd $RUN && python $HERE/bench_gemm.py" > gemm.log 2>&1; echo "exit $?"; grep -E "SUMMARY|^  \(" gemm.log | tail -20; fi
step "fa d128"; timeout 15m $R "cd $RUN && python $HERE/run_fa.py $TL_SRC --dim 128 --seq_len 4096 --heads 8" > fa128.log 2>&1; echo "exit $?"; tail -15 fa128.log
step "fa d256 causal gqa"; timeout 15m $R "cd $RUN && python $HERE/run_fa.py $TL_SRC --dim 256 --seq_len 4096 --heads 24 --groups 6 --causal" > fa256.log 2>&1; echo "exit $?"; tail -15 fa256.log
step "amd unit tests"; cp $TL_SRC/testing/python/amd/test_tilelang_{gemm_mfma_intrinsic,test_amd,hip_codegen,hip_gemm_block_size}.py $RUN/; cp $HERE/conftest_rocm_only.py $RUN/conftest.py; timeout 20m $R "cd $RUN && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider -rfE --rootdir=$RUN test_tilelang_gemm_mfma_intrinsic.py test_tilelang_test_amd.py test_tilelang_hip_codegen.py test_tilelang_hip_gemm_block_size.py" > amdtests.log 2>&1; echo "exit $?"; tail -30 amdtests.log
step "gdn compile test"; mkdir -p $RUN/gdn && cp $TL_SRC/examples/gdn/*.py $RUN/gdn/; cp $HERE/conftest_rocm_only.py $RUN/gdn/conftest.py; timeout 15m $R "cd $RUN/gdn && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q -p no:cacheprovider -rfE --rootdir=$RUN/gdn test_example_gdn_compilation.py" > gdn.log 2>&1; echo "exit $?"; tail -30 gdn.log
step "done"
