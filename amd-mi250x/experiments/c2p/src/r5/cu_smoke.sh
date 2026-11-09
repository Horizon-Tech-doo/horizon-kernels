#!/bin/bash
# Login node, inside the image, CPU only. Point a shadow root at the CUDA build so tilelang loads it, not fork-main/build.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
R=$E/r5; P=$R/py; rm -rf $P; mkdir -p $P; ln -s $F/tilelang $P/tilelang; ln -s $R/build-cuda $P/build; ln -s $F/3rdparty $P/3rdparty; ln -s $F/testing $P/testing; ln -s $F/examples $P/examples
TK=$R/tk/nvidia/cu13
singularity exec -B "$TL_BIND" "$IMG" bash -c "export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$R/cache/smoke PATH=$TK/bin:\$PATH LD_LIBRARY_PATH=$R/build-cuda/lib:$TK/lib:\$LD_LIBRARY_PATH F=$F; cd $P; python3 -I -c 'import sys,runpy; sys.path[:0]=[\"$P\",\"$VENV\"]; sc=\"$R/cu_smoke.py\"; runpy.run_path(sc, run_name=\"__main__\")'" > $R/cu_smoke.txt 2>&1
singularity exec -B "$TL_BIND" "$IMG" bash -c "export CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 TILELANG_CACHE_DIR=$R/cache/smoke PATH=$TK/bin:\$PATH LD_LIBRARY_PATH=$R/build-cuda/lib:$TK/lib:\$LD_LIBRARY_PATH; cd $P; timeout 200 python3 -I -c 'import sys; sys.path[:0]=[\"$P\",\"$VENV\"]; import pytest, tilelang.testing as tt; tt.requires_cuda = (lambda f: f); sys.exit(pytest.main(sys.argv[1:]))' -q --no-header -p no:cacheprovider --tb=line -k \"not nonexistent\" --rootdir=$P testing/python/transform/test_tilelang_transform_materialize_ws_schedule.py" > $R/cu_ws.txt 2>&1
rm -rf $P $R/cache/smoke; echo SMOKE_DONE >> $R/cu_smoke.txt
