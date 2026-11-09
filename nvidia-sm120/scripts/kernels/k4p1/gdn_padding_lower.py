# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Lower the padding-tail candidate without initializing or loading a CUDA module."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TVM_FFI_DISABLE_TORCH_C_DLPACK'] = '1'
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
from tvm.target import Target
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler

spec = importlib.util.spec_from_file_location('padding_segment', sys.argv[1])
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
for hw in (0, 2):
    wm, splits = sg.bucket(4)
    jf = sg.segment_kernel
    tir = jf.get_tir(376, wm, *splits, sg.SK_CFG[0], sg.SK_CFG[1], 4, 1, 1e-6, 1, 0,
                     2_000_000_000, 0, 0, sg.NQ, hw, gdn=1)
    with target:
        with tilelang.transform.PassContext(config=jf.pass_configs or {}):
            artifact = tilelang.lower(tir, target=target)
    cu = out / f'padding_hw{hw}.cu'
    cu.write_text(artifact.kernel_source)
    result = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w',
                             '-I'+TILELANG_TEMPLATE_PATH, '-I'+CUTLASS_INCLUDE_DIR,
                             '-Xptxas', '-v', '-o', str(cu.with_suffix('.cubin')), str(cu)],
                            capture_output=True, text=True)
    (out / f'padding_hw{hw}.compiler.log').write_text(result.stdout + result.stderr)
    for line in (result.stdout + result.stderr).splitlines():
        if any(word in line for word in ('registers', 'spill', 'error')):
            print(line, flush=True)
    result.check_returncode()
    print(f'PADDING_TAIL_COMPILED hw={hw}', flush=True)
