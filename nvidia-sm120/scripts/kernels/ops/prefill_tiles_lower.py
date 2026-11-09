# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compile prefill tile candidates without initializing a CUDA context."""
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

spec = importlib.util.spec_from_file_location('prefill_candidate', sys.argv[1])
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
for tq, bn in ((16, 32), (8, 32), (8, 64)):
    # HND interleaved K/V: pages, heads=4, page=64, 2*D=512.
    jf = sg.attn_prefill_kernel_v3
    tir = jf.get_tir(4, 6, 256, 64, tq, bn, 131072, 512, 32768)
    with target:
        with tilelang.transform.PassContext(config=jf.pass_configs or {}):
            artifact = tilelang.lower(tir, target=target)
    cu = out / f'prefill_q{tq}_kv{bn}.cu'
    cu.write_text(artifact.kernel_source)
    result = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w',
                             '-I'+TILELANG_TEMPLATE_PATH, '-I'+CUTLASS_INCLUDE_DIR,
                             '-Xptxas', '-v', '-o', str(cu.with_suffix('.cubin')), str(cu)],
                            capture_output=True, text=True)
    (out / f'prefill_q{tq}_kv{bn}.compiler.log').write_text(result.stdout + result.stderr)
    for line in (result.stdout + result.stderr).splitlines():
        if any(word in line for word in ('registers', 'spill', 'error')):
            print(line, flush=True)
    result.check_returncode()
    print(f'PREFILL_COMPILED tq={tq} bn={bn}', flush=True)
