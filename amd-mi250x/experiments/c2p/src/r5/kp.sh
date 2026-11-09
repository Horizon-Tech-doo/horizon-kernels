#!/bin/bash
# Login node, CPU only: the actual collection error of the 4 kpool test files on the series tree.
source /scratch/project_XXXXXXXXX/tilelang/exp/C2P/src/cenv.sh
singularity exec -B "$TL_BIND" "$TL_SIF" bash -c "\$WITH_CONDA; source $TL_ROOT/venv/bin/activate; export ROCM_PATH=$TL_ROOT/hipcc-wrap PYTHONNOUSERSITE=1 PYTHONPATH=$F:$S LD_LIBRARY_PATH=$F/build/lib:\$LD_LIBRARY_PATH; cd $F; timeout 100 python -m pytest --co -q --no-header -p no:cacheprovider --tb=short testing/python/amd/test_tilelang_hip_glm53_kpool_compress.py 2>&1 | tail -22; python -c \"import examples;print(examples.__path__)\" 2>&1 | tail -1" | cut -c1-170
