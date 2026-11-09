# Shared settings for the TileLang-on-LUMI scripts. Source, do not execute.
export PROJ=project_XXXXXXXXX
export TL_ROOT=/scratch/$PROJ/tilelang
export TL_SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif
export TL_VERSION=0.1.15            # PyPI fat wheel (CUDA+ROCm), = git tag v0.1.15 a35f8ddf45eba16c21211ec8822d56ce5363036f
export TL_SRC=$TL_ROOT/src-v0.1.15  # examples/tests matching the wheel
# Bind both the /scratch symlink and its /pfs target, otherwise venv paths break inside the container.
export TL_BIND=/pfs,/scratch,/flash
export PIP_CACHE_DIR=$TL_ROOT/pipcache
export PYTHONNOUSERSITE=1           # never fall back to ~/.local (shared with other jobs)
