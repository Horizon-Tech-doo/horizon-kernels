# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compile model-derived segment shapes and legacy control without GPU loading."""
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

spec = importlib.util.spec_from_file_location('resource_segment', sys.argv[1])
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)
task_spec = importlib.util.spec_from_file_location('candidate_tasks', str(Path(sys.argv[1]).with_name('tasks.py')))
tasks = importlib.util.module_from_spec(task_spec)
task_spec.loader.exec_module(tasks)
sg.tk = tasks
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
cases = [('legacy', (5120,17408,6144), 16384, -1, True, 2)]
cases += [(f'generic_hw{hw}', (4096,14336,4096), 12288, 80, False, hw) for hw in (0,1,2)]
cases += [('generic_no_aux', (4096,14336,4096), 16384, 0, False, 2)]
cases += [(f'generic_m32_hw{hw}', (4096,14336,4096), 12288, 80, False, hw) for hw in (0,2)]
if len(sys.argv) > 3 and sys.argv[3] == 'large-only':
    cases = cases[-2:]
for tag, geometry, nq, nba, centered, hw in cases:
    m = 32 if '_m32_' in tag else 4
    # Explicit split factors exercise GU split-K independently of heuristic selection.
    splits = (4,1,8,4) if tag == 'legacy' else (4,2,8,4)
    sg.validate_splits(geometry, splits)
    sc = sg.build_schedule(m,376,splits,sg.SK_CFG,nq,geometry=geometry,nba=nba)
    sc.check()
    print('SCHEDULE', tag, sc.stats(), flush=True)
    jf = sg.segment_kernel
    tir = jf.get_tir(376,16 if m <= 16 else 32,*splits,sg.SK_CFG[0],sg.SK_CFG[1],4,1,1e-6,1,0,
                     2_000_000_000,0,0,nq,hw,geometry=geometry,nba=nba,zero_centered_gamma=centered)
    with target:
        with tilelang.transform.PassContext(config=jf.pass_configs or {}):
            artifact = tilelang.lower(tir, target=target)
    cu = out / f'{tag}.cu'
    cu.write_text(artifact.kernel_source)
    result = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20', '-w',
                             '-I'+TILELANG_TEMPLATE_PATH, '-I'+CUTLASS_INCLUDE_DIR,
                             '-Xptxas', '-v', '-o', str(cu.with_suffix('.cubin')), str(cu)],
                            capture_output=True, text=True)
    (out / f'{tag}.compiler.log').write_text(result.stdout + result.stderr)
    for line in (result.stdout + result.stderr).splitlines():
        if any(word in line for word in ('registers', 'spill', 'error')):
            print(line, flush=True)
    result.check_returncode()
    print(f'SEGMENT_GEOMETRY_COMPILED {tag}', flush=True)
