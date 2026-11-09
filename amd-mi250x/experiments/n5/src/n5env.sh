# Source in jobs and on the login node. TL_L runs in the LUMI stack (ROCm 6.2.4, clang 18), TL_I in the vLLM image (ROCm 7.2.3, clang 22), both with TileLang 0.1.15 from our venv.
E=/scratch/project_XXXXXXXXX/tilelang/exp/N5; S=$E/src
source /scratch/project_XXXXXXXXX/tilelang/scripts/env.sh
IMG=/flash/project_XXXXXXXXX/vllm-check/images/vllm-rocm-v0.30.0.sif
VENV=$TL_ROOT/venv/lib/python3.12/site-packages
# TileLang's cache key does not see the patch: each flavour + patch.py version gets its own cache (job 23503844's PY arms reused kernels from an older patch.py).
PH=$(md5sum $S/patch.py | cut -c1-6)
# $1 flavour, $2 python script + args
TL_L() { singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache/L$1${1:+_$PH} TRITON_CACHE_DIR=$E/cache/L$1${1:+_$PH}/triton N5_PATCH=$1 N5_STACK=lumi $N5_ENV; cd $S; python $2"; }
TL_I() { local sc=${2%% *}; local ar=${2#* }; [ "$ar" = "$2" ] && ar=""; singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$E/cache/I$1${1:+_$PH} TRITON_CACHE_DIR=$E/cache/I$1${1:+_$PH}/triton HF_HUB_OFFLINE=1 N5_PATCH=$1 N5_STACK=image $N5_ENV; cd $S; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$S\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $sc $ar"; }
