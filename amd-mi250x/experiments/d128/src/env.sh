# Track D128, lever 2. Stacks are N5 round 5's image stack. FM: fork-main on the vLLM image (clang 22), libtilelang 63bb886f. WH: same image, stock wheel 0.1.15. Own caches and source copies only; fork-main, C2P, N5 and scripts/ are read-only.
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src5/n5env5.sh
D=$TL_ROOT/exp/D128; SRC=$D/src
_run() { # $1 cache dir, $2 extra sys.path entry or -, $3 LD prefix or -, $4 script + args
  local xp="" ld=""; [ "$2" != "-" ] && xp="sys.path.insert(0,\"$2\");"; [ "$3" != "-" ] && ld="LD_LIBRARY_PATH=$3:\$LD_LIBRARY_PATH"
  singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TILELANG_CACHE_DIR=$1 TRITON_CACHE_DIR=$1/triton HF_HUB_OFFLINE=1 $ld; cd $SRC; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VENV\"); $xp sys.path.insert(0,\"$SRC\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $4"
}
FM() { _run "$1" "$F" "$F/build/lib" "$2"; }
WH() { _run "$1" - - "$2"; }
