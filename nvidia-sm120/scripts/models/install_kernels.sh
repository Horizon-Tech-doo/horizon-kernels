# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
export UV_CACHE_DIR=$HK_DATA/.uvcache LD_LIBRARY_PATH=$HK_DATA/cuda-compat
export CUDA_HOME=$HK_VENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$HK_VENV/bin:$PATH
export TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=32 CAUSAL_CONV1D_FORCE_BUILD=TRUE PYTHONPATH=$HK_DATA/llmc-pkgs
U=~/.local/bin/uv; PY=$HK_VENV/bin/python
$U pip install --no-deps --target $HK_DATA/llmc-pkgs --python $PY flash-linear-attention fla-core einops 2>&1 | tail -3
$U pip install --no-deps --no-build-isolation --target $HK_DATA/llmc-pkgs --python $PY causal-conv1d 2>&1 | tail -5
$PY -c "import fla, causal_conv1d, torch; from causal_conv1d import causal_conv1d_fn; x=torch.randn(1,64,32,device=\"cuda\",dtype=torch.bfloat16); w=torch.randn(64,4,device=\"cuda\",dtype=torch.bfloat16); print(\"KERNELS OK\", causal_conv1d_fn(x,w).shape, fla.__version__)"
