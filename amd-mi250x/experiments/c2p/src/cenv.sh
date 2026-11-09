# Source in jobs and on the login node. fork-main is upstream main + the series. FM_L: LUMI stack (ROCm 6.2.4, clang 18). FM_I: vLLM image (ROCm 7.2.3, clang 22).
E=/scratch/project_XXXXXXXXX/tilelang/exp/C2P; S=$E/src
source /scratch/project_XXXXXXXXX/tilelang/scripts/env.sh
IMG=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VENV=$TL_ROOT/venv/lib/python3.12/site-packages
F=$TL_ROOT/fork-main
FM_ENV=${FM_ENV:-}
FM_L() { singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$F:$S TILELANG_CACHE_DIR=$E/cache/L LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH $FM_ENV; cd $E; python $1"; }
FM_I() { local sc=${1%% *}; local ar=${1#* }; [ "$ar" = "$1" ] && ar=""; singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache/I TRITON_CACHE_DIR=$E/cache/I/triton HF_HUB_OFFLINE=1 LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH $FM_ENV; cd $E; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$F\"); sys.path.insert(0,\"$S\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $sc $ar"; }
# Stock wheel, image stack: hand-edited N5 kernels as the comparator.
WH_I() { local sc=${1%% *}; local ar=${1#* }; [ "$ar" = "$1" ] && ar=""; singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache/Iw TRITON_CACHE_DIR=$E/cache/Iw/triton HF_HUB_OFFLINE=1 $FM_ENV; cd $E; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$S\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $sc $ar"; }
WH_L() { singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$S TILELANG_CACHE_DIR=$E/cache/Lw $FM_ENV; cd $E; python $1"; }
