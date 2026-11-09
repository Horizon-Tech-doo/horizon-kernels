# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Task-level timing diagnosis, not a performance or deployment gate.

Four unique weight sets exceed L2. Instrumentation changes instruction streams;
raw globaltimer stamps diagnose phases but do not prove a speed improvement.
"""
import gzip
import json
from pathlib import Path
import ctypes
import sys
import torch
import tilelang.contrib

root, output = map(Path, sys.argv[1:3])
tilelang.contrib.__path__.insert(0, str(root/'contrib'))
from tilelang.contrib.horizon_k4.mlp_mixed_trace import TraceMixedMlp
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl, decode_regfrag as rf
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from bench_mlp_l2_05_10_helper import summarize_trace

def split(n,k,rows):
    c=xl.config(n,k,rows)
    return (tuple(c[1:]) if len(c)>1 else rf.config(n,k,rows))[3] if c[0]=='rf' else c[5]

def capture(fn):
    fn(); torch.cuda.synchronize()
    for attempt in range(4):
        graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph,capture_error_mode='thread_local'): result=fn()
            return graph,result
        except RuntimeError:
            torch.cuda.synchronize()
            if attempt==3: raise

out=output.open('x')
raw=gzip.open(output.with_suffix('.timestamps.jsonl.gz'),'xt')
def emit(r):
    line=json.dumps(r);out.write(line+'\n');out.flush();print(line,flush=True)

props=torch.cuda.get_device_properties(0)
from tilelang.contrib.horizon_k4.runtime import _cudart
cudart=_cudart();cudart.cudaDeviceGetAttribute.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
regcap=ctypes.c_int();assert cudart.cudaDeviceGetAttribute(ctypes.byref(regcap),82,0)==0
assert regcap.value>0
compiled=json.loads((root/'offline/result.json').read_text())['records']
registers={(r['rows'],r['block_n'],r['stages'],r['grid']):max(map(int,r['registers'])) for r in compiled}
layers=[]
for layer in range(4):
    pair=[]
    for j,(n,k) in enumerate(((34816,5120),(5120,17408))):
        data=synthetic_layer(n,k,seed=1042026+layer*10+j)
        pair.append(xl.prepare(*data));del data
    layers.append(pair)
storages={}
for pair in layers:
    for prep in pair:
        for key in ('Wg','Sg','alpha'):
            storage=getattr(prep,key).untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
assert sum(storages.values())>512*1024**2
emit(dict(test='working_set',layers=4,weight_bytes=sum(storages.values()),sms=props.multi_processor_count))
separator=torch.empty(256*1024**2,dtype=torch.uint8,device='cuda')
for rows in (4,32):
    x=torch.randn(rows,5120,dtype=torch.bfloat16,device='cuda')
    activate=torch.compile(lambda z:torch.nn.functional.silu(z[:,:17408])*z[:,17408:],fullgraph=True,dynamic=False)
    def reference():
        value=x;values=[]
        for gate,down in layers:
            gu=xl.linear(value,gate);a=activate(gu);value=xl.linear(a,down)
            values.append((gu,a,value))
        return values
    base,expected=capture(reference)
    for width,stages,grid in ((128,6,188),(256,3,188),(256,4,188)):
        occupancy=-(-grid//props.multi_processor_count)
        wm=16 if rows<=16 else 32
        shared=max(stages,4)*wm*64*2+max(stages*(width//32),16)*2*576+1024
        assert occupancy*shared<=props.shared_memory_per_multiprocessor
        regs=registers[(rows,width,stages,grid)]
        assert ((regs+15)//16)*16*max(width,128)*occupancy<=regcap.value
        blocks=[TraceMixedMlp(rows,5120,17408,(split(34816,5120,rows),split(5120,17408,rows)),grid,
                             stages=stages,block_n=width) for _ in layers]
        def chain():
            value=x
            for block,(gate,down) in zip(blocks,layers):value=block(value,gate,down)
            return value
        graph,result=capture(chain)
        for replay in range(8):
            x.normal_().mul_(1 if replay%2==0 else 256)
            base.replay();graph.replay();torch.cuda.synchronize()
            for block,refs in zip(blocks,expected):
                block.check()
                assert all(torch.isfinite(t).all().item() and torch.equal(t,r)
                           for t,r in zip((block.gu,block.a,block.y),refs))
        emit(dict(test='profile_correctness',rows=rows,width=width,stages=stages,grid=grid,replays=8,exact=True))
        for trial in range(3):
            for block in blocks:block.trace.zero_()
            separator.zero_();graph.replay();torch.cuda.synchronize()
            for layer,block in enumerate(blocks):
                block.check();trace=block.trace.cpu().tolist()
                summary=summarize_trace(trace,17408,width,5120,128,split(5120,17408,rows),props.multi_processor_count)
                emit(dict(test='task_profile',rows=rows,width=width,stages=stages,grid=grid,
                          trial=trial,layer=layer,summary=summary))
                raw.write(json.dumps(dict(rows=rows,width=width,stages=stages,grid=grid,
                                          trial=trial,layer=layer,trace=trace))+'\n');raw.flush()
        del blocks,graph
raw.close()
emit(dict(test='complete',status='instrumented diagnostic; no speed or serving claim'))
