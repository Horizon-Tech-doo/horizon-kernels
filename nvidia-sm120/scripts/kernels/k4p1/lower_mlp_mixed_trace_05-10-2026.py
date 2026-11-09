# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Offline SM120a lowering/compilation: never loads a CUDA module or opens a context."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
torch.cuda.is_available = lambda: False
torch.cuda.device_count = lambda: 0
import tilelang
import tilelang.contrib
root, output = map(Path, sys.argv[1:3])
tilelang.contrib.__path__.insert(0, str(root/'contrib'))
from tilelang.contrib.horizon_k4.mlp import mlp_kernel
from tilelang.contrib.horizon_k4.mlp_static_plan import mlp_static_kernel
from tilelang.contrib.horizon_k4.mlp_mixed_trace import mlp_mixed_trace_kernel
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl, decode_regfrag as rf
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
from tvm.target import Target


def split(n, k, rows):
    config = xl.config(n, k, rows)
    return (tuple(config[1:]) if len(config) > 1 else rf.config(n, k, rows))[3] if config[0] == 'rf' else config[5]


output.mkdir(exist_ok=False)
target = Target({'kind': 'cuda', 'arch': 'sm_120a'})
records = []
for rows in (4,32):
    sg, sd = split(34816, 5120, rows), split(5120, 17408, rows)
    for block_n,stages,grid in ((128,6,188),(256,3,188),(256,4,188)):
        stages_down,block_n_down=4,128
        block_k,plan=64,block_n==256
        hw, fused = 1, 0
        begin = time.time()
        tir = mlp_mixed_trace_kernel.get_tir(
            5120,17408,16 if rows<=16 else 32,sg,sd,grid,rows,stages,block_k=block_k,block_n=block_n,stages_down=stages_down,block_n_down=block_n_down)
        with target:
            with tilelang.transform.PassContext(config=mlp_kernel.pass_configs or {}):
                artifact = tilelang.lower(tir, target=target)
        source = artifact.kernel_source
        path = output/f'mlp-m{rows}-n{block_n}-stages{stages}-grid{grid}.cu'
        path.write_text(source)
        process = subprocess.run([get_nvcc_compiler(), '--cubin', '-O3', '-arch=sm_120a', '-std=c++20',
            '-w', '-I'+TILELANG_TEMPLATE_PATH, '-I'+CUTLASS_INCLUDE_DIR, '-Xptxas', '-v',
            '-o', str(path.with_suffix('.cubin')), str(path)], capture_output=True, text=True, timeout=240)
        path.with_suffix('.log').write_text(process.stdout+process.stderr)
        record = dict(stages_down=stages_down,block_n_down=block_n_down,block_n=block_n,grid=grid,rows=rows, hw=hw, fused=fused, stages=stages, plan=plan, block_k=block_k, splits=[sg, sd], returncode=process.returncode,
            registers=re.findall(r'Used (\d+) registers', process.stderr),
            spill_stores=re.findall(r'(\d+) bytes spill stores', process.stderr),
            spill_loads=re.findall(r'(\d+) bytes spill loads', process.stderr),
            seconds=round(time.time()-begin, 1), kernel_parameters=artifact.params)
        assert process.returncode == 0 and all(int(x)==0 for x in record['spill_stores']+record['spill_loads']), record
        assert record['registers'] and max(map(int,record['registers'])) <= 256, record
        records.append(record)
        print(json.dumps(record, default=str), flush=True)
        if process.returncode:
            raise RuntimeError(process.stderr[-2000:])
(output/'result.json').write_text(json.dumps(dict(tilelang_version=tilelang.__version__, records=records),
                                        indent=2, default=str)+'\n')
print('MLP_OFFLINE_COMPILE_PASS', flush=True)
