# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only production/isolated-K4 CUDA and PTX comparison; no runtime timing."""
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
def ptx_op_counts(ptx):
    counts=collections.Counter()
    for line in ptx.splitlines():
        match=re.match(r'\s*(?:@!?%?[A-Za-z_][A-Za-z_0-9]*\s+)?([a-z][a-z0-9_]*(?:\.[a-zA-Z0-9_:]+)*)(?:\s|;)',line)
        if match:counts[match[1]]+=1
    return dict(sorted(counts.items()))

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import torch
torch.cuda.is_available=lambda:False
torch.cuda.device_count=lambda:0
import tilelang
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl, decode_regfrag as rf
from tilelang.cuda.backend import TILELANG_TEMPLATE_PATH,CUTLASS_INCLUDE_DIR
from tilelang.contrib.nvcc import get_nvcc_compiler
from tvm.target import Target
output=Path(sys.argv[1]);output.mkdir(exist_ok=False)
old=Path(os.environ.get("HK_WORK", ".")) / 'k4/candidates/mlp-v47-05-10-2026/offline'
target=Target({'kind':'cuda','arch':'sm_120a'})
records=[]
def inspect(path,meta,build_cubin=False):
    command=[get_nvcc_compiler(),'-O3','-arch=sm_120a','-std=c++20','-w','-I'+TILELANG_TEMPLATE_PATH,'-I'+CUTLASS_INCLUDE_DIR]
    for mode in (['cubin','ptx'] if build_cubin else ['ptx']):
        process=subprocess.run(command+['--'+mode,'-Xptxas','-v','-o',str(path.with_suffix('.'+mode)),str(path)],capture_output=True,text=True,timeout=180)
        path.with_suffix('.'+mode+'.log').write_text(process.stdout+process.stderr)
        assert process.returncode==0,process.stderr
    ptx=path.with_suffix('.ptx').read_text()
    ops=ptx_op_counts(ptx)
    assert ops, path
    log=path.with_suffix('.cubin.log').read_text() if build_cubin else path.with_suffix('.existing.log').read_text()
    rec=dict(meta, cuda_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
        registers=re.findall(r'Used (\d+) registers',log),spill_stores=re.findall(r'(\d+) bytes spill stores',log),spill_loads=re.findall(r'(\d+) bytes spill loads',log),static_ptx_ops=dict(sorted(ops.items())))
    assert rec['registers'] and all(int(v)==0 for v in rec['spill_stores']+rec['spill_loads'])
    records.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='static_ptx_ops'}),flush=True)
for n,k in ((34816,5120),(5120,17408)):
    for rows in (4,32):
        cfg=xl.config(n,k,rows)
        if cfg[0]=='rf':
            assert not xl._rf_padded(rows)
            args=tuple(cfg[1:]) if len(cfg)>1 else rf.config(n,k,rows)
            factory=rf.nvfp4_regfrag_kernel; family='rf'
            width=32*args[1]; stages=args[4]; threads=32*args[1]
        else:
            args=cfg;factory=xl.nvfp4_regfrag_xl_kernel;family='xl'
            width=32*args[2]*args[3]; stages=args[6];threads=32*args[0]*args[2]
        tir=factory.get_tir(n,k,*args)
        with target:
            with tilelang.transform.PassContext(config=factory.pass_configs or {}):
                artifact=tilelang.lower(tir,target=target)
        path=output/f'production-n{n}-m{rows}.cu';path.write_text(artifact.kernel_source)
        inspect(path,dict(name=path.stem,family=family,n=n,k=k,rows=rows,cfg=args,width=width,stages=stages,threads=threads),True)
        candidates=list(old.glob(f'gemm-n{n}-m{rows}-n{width}-s{stages}-t{threads}-g188.cu'))
        if not candidates:
            candidates=list(old.glob(f'gemm-n{n}-m{rows}-n128-s4-t128-g188.cu'))
        assert len(candidates)==1,candidates
        src=candidates[0];path=output/(src.name);path.write_bytes(src.read_bytes());path.with_suffix('.existing.log').write_bytes(src.with_suffix('.log').read_bytes())
        inspect(path,dict(name=path.stem,family='isolated_k4_v47',n=n,k=k,rows=rows,matched_geometry=(f'n{width}-s{stages}-t{threads}-' in src.name)))
report={'scope':'Static PTX only, not SASS scheduling or executed instruction counts. Isolated K4 v47 predates vector4 reduction; dynamic-M production differs from fixed-M K4. No timing or quality proof.', 'gpu_context_created':False,'records':records}
(output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
print('GEMM_STATIC_PTX_CPU_PASS',flush=True)
