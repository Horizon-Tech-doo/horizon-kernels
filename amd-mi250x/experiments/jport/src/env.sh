# JPORT. Wheel 0.1.15 on the vLLM image (clang 22), the landed module's runtime. Own caches and source copies; scripts/, D128 and fork-main read-only.
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
X=$TL_ROOT/exp/JPORT; SRC=$X/src
WH() { # $1 cache dir, $2 module dir, $3 script + args
  singularity exec -B "$TL_BIND" "$IMG" bash -c "export VLLM_NO_USAGE_STATS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TILELANG_CACHE_DIR=$1 TRITON_CACHE_DIR=$1/triton HF_HUB_OFFLINE=1; cd $SRC; python3 -I -c 'import sys,runpy; sys.path.insert(0,\"$VENV\"); sys.path.insert(0,\"$SRC\"); sys.path.insert(0,\"$2\"); sc=sys.argv[1]; sys.argv=sys.argv[1:]; runpy.run_path(sc, run_name=\"__main__\")' $3"
}
nodetag() { echo "node_tag $( (cat $TL_ROOT/exp/N5/.salt; hostname) | md5sum | cut -c1-6)"; }
