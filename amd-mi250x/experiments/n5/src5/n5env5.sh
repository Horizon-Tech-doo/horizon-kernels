# Round 5: round 4's stacks with this round's source folder and cache. FM5_L: fork-main on clang 18, C2P's c3.py and tlfa.py read-only, own cache.
source /scratch/project_XXXXXXXXX/tilelang/exp/N5/src/n5env.sh
S=$E/src5; PH=r5
N5_ENV=""
F=$TL_ROOT/fork-main; C=$TL_ROOT/exp/C2P/src
FM5_L() { singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$F:$C TILELANG_CACHE_DIR=$E/cache/Lc2p LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH C3_STACK=lumi $FM_ENV; cd $S; python $C/c3.py $1"; }
nodetag() { echo "node_tag $( (cat $E/.salt; hostname) | md5sum | cut -c1-6)"; }
