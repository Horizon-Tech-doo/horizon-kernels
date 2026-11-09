# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only lower/compile of persistent RMSNorm macro; no CUDA module load."""
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
import tilelang.language as T
from tvm.target import Target
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler

spec = importlib.util.spec_from_file_location('portable_tasks', sys.argv[1])
tk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tk)
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)

@tilelang.jit
def kernel(H: int, residual: bool, centered: bool):
    vt, _ = tk.norm_geometry(H)
    norm = tk.norm_task_body(H, 1e-6, residual, zero_centered_gamma=centered)
    @T.prim_func
    def main(X: T.Tensor((4, H), T.bfloat16), R: T.Tensor((4, H), T.bfloat16),
             W: T.Tensor((H,), T.bfloat16), Y: T.Tensor((4, H), T.bfloat16),
             RO: T.Tensor((4, H), T.bfloat16)):
        with T.Kernel(1, threads=128):
            tid = T.get_thread_binding(0)
            red = T.alloc_shared((vt // 32,), T.float32)
            tot = T.alloc_shared((1,), T.float32)
            for row in T.serial(4):
                norm(X, R, W, Y, RO, row, tid, red, tot)
    return main

target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
for h in (256, 768, 4096, 5120):
    for residual in (False, True):
        for centered in (False, True):
            tir = kernel.get_tir(h, residual, centered)
            with target:
                with tilelang.transform.PassContext(config=kernel.pass_configs or {}):
                    artifact = tilelang.lower(tir, target=target)
            tag = f'h{h}_r{int(residual)}_g{int(centered)}'
            cu = out / (tag + '.cu'); cu.write_text(artifact.kernel_source)
            result = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w',
                                     '-I'+TILELANG_TEMPLATE_PATH, '-I'+CUTLASS_INCLUDE_DIR,
                                     '-Xptxas', '-v', '-o', str(cu.with_suffix('.cubin')), str(cu)],
                                    capture_output=True, text=True)
            (out / (tag + '.log')).write_text(result.stdout + result.stderr)
            result.check_returncode()
            print(tag, ':', '; '.join(s.strip() for s in result.stderr.splitlines()
                                      if 'registers' in s or 'spill' in s), flush=True)
print('PERSISTENT_NORM_COMPILE_PASS 16', flush=True)
