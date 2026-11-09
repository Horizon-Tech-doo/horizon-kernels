# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compile the intermediate persistent-grid candidates without a CUDA context."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import torch
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
import tilelang
import tilelang.contrib
root,output=map(Path,sys.argv[1:3])
tilelang.contrib.__path__.insert(0,str(root/'contrib'))
from tilelang.contrib.horizon_k4.mlp_mixed import mlp_mixed_kernel
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl,decode_regfrag as rf
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH,CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
from tvm.target import Target

def split(n,k,rows):
    cfg=xl.config(n,k,rows)
    return (tuple(cfg[1:]) if len(cfg)>1 else rf.config(n,k,rows))[3] if cfg[0]=='rf' else cfg[5]

output.mkdir(exist_ok=False)
target=Target({'kind':'cuda','arch':'sm_120a'})
records=[]
def compile_one(factory,args,kwargs,meta):
    begin=time.time()
    tir=factory.get_tir(*args,**kwargs)
    with target:
        with tilelang.transform.PassContext(config=factory.pass_configs or {}):
            artifact=tilelang.lower(tir,target=target)
    path=output/(meta['name']+'.cu')
    path.write_text(artifact.kernel_source)
    process=subprocess.run([get_nvcc_compiler(),'--cubin','-O3','-arch=sm_120a','-std=c++20',
        '-w','-I'+TILELANG_TEMPLATE_PATH,'-I'+CUTLASS_INCLUDE_DIR,'-Xptxas','-v',
        '-o',str(path.with_suffix('.cubin')),str(path)],capture_output=True,text=True,timeout=240)
    path.with_suffix('.log').write_text(process.stdout+process.stderr)
    record=dict(meta,returncode=process.returncode,
        registers=re.findall(r'Used (\d+) registers',process.stderr),
        spill_stores=re.findall(r'(\d+) bytes spill stores',process.stderr),
        spill_loads=re.findall(r'(\d+) bytes spill loads',process.stderr),
        seconds=round(time.time()-begin,1),kernel_parameters=artifact.params)
    print(json.dumps(record,default=str),flush=True)
    assert process.returncode==0 and record['registers'],record
    assert all(int(x)==0 for x in record['spill_stores']+record['spill_loads']),record
    records.append(record)

for rows in (1,4,8,16,32):
    sg,sd=split(34816,5120,rows),split(5120,17408,rows)
    for grid in (188,236,272,320,376):
        compile_one(mlp_mixed_kernel,(5120,17408,16 if rows<=16 else 32,sg,sd,grid,rows,4),
            dict(block_k=64,block_n=128,stages_down=4,block_n_down=128,activation_pair=True),
            dict(name=f'mlp-m{rows}-s4-grid{grid}',rows=rows,stages=4,
                 staged_init_gu=False,staged_init_down=False,grid=grid,block_n=128,splits=[sg,sd]))
(output/'result.json').write_text(json.dumps(dict(tilelang_version=tilelang.__version__,records=records),indent=2,default=str)+'\n')
print('MLP_GRID_SWEEP_OFFLINE_COMPILE_PASS',flush=True)
