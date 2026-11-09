# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Uninstrumented operation isolation; diagnostic, never an MLP speed gate.

Each operation's distinct weight sets exceed L2, including the smaller DOWN.
Compiler liveness and missing dependency/reset work differ from the combined MLP.
"""
import ctypes,json,statistics,sys
from pathlib import Path
import torch
import tilelang.contrib
from isolated_gemm_configs_05_10_helper import configs,label
root,output=map(Path,sys.argv[1:3])
tilelang.contrib.__path__.insert(0,str(root/'contrib'))
from tilelang.contrib.horizon_k4.gemm_isolated_probe import IsolatedGemm
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl,decode_regfrag as rf
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_k4.runtime import _cudart

def split(n,k,rows):
    c=xl.config(n,k,rows)
    return (tuple(c[1:]) if len(c)>1 else rf.config(n,k,rows))[3] if c[0]=='rf' else c[5]
def capture(fn):
    fn();torch.cuda.synchronize()
    for attempt in range(4):
        graph=torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph,capture_error_mode='thread_local'):result=fn()
            return graph,result
        except RuntimeError:
            torch.cuda.synchronize()
            if attempt==3:raise
out=output.open('x')
def emit(r):
    s=json.dumps(r);out.write(s+'\n');out.flush();print(s,flush=True)
props=torch.cuda.get_device_properties(0)
cudart=_cudart();cudart.cudaDeviceGetAttribute.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
regcap=ctypes.c_int();assert cudart.cudaDeviceGetAttribute(ctypes.byref(regcap),82,0)==0
compiled=json.loads((root/'offline/result.json').read_text())['records']
registers={(r['n'],r['rows'],label(r)):max(map(int,r['registers'])) for r in compiled}
separator=torch.empty(256*1024**2,device='cuda',dtype=torch.uint8)
for n,k,count in ((34816,5120,6),(5120,17408,12)):
    weights=[]
    for i in range(count):
        data=synthetic_layer(n,k,seed=1052026+n+i)
        weights.append(xl.prepare(*data));del data
    storage={}
    for w in weights:
        for attr in ('Wg','Sg','alpha'):
            s=getattr(w,attr).untyped_storage();storage[s.data_ptr()]=s.nbytes()
    assert sum(storage.values())>512*1024**2
    emit(dict(test='working_set',n=n,k=k,weights=count,weight_bytes=sum(storage.values())))
    for rows in (4,32):
        sp=split(n,k,rows)
        x=torch.randn(rows,k,device='cuda',dtype=torch.bfloat16)
        def reference():return [xl.linear(x,w) for w in weights]
        base,refs=capture(reference)
        graphs={'baseline':base};objects={}
        for cfg in configs(rows,n):
            name=label(cfg);regs=registers[(n,rows,name)]
            wm=16 if rows<=16 else 32
            occupancy=-(-cfg['grid']//props.multi_processor_count)
            shared=cfg['stages']*(wm*64*2+cfg['width']//32*2*576)+1024
            assert shared*occupancy<=props.shared_memory_per_multiprocessor
            assert ((regs+15)//16)*16*cfg['threads']*occupancy<=regcap.value
            probes=[IsolatedGemm(n,k,rows,sp,**cfg) for _ in weights]
            def launch(probes=probes):return [p(x,w) for p,w in zip(probes,weights)]
            graph,outputs=capture(launch)
            for replay in range(64):
                x.normal_().mul_(1 if replay%2==0 else 256)
                base.replay();graph.replay();torch.cuda.synchronize()
                for p,y,r in zip(probes,outputs,refs):
                    p.check();assert torch.isfinite(y).all().item() and torch.isfinite(r).all().item()
                    assert torch.equal(y,r),(n,rows,name,replay,'identity')
            emit(dict(test='correctness',n=n,k=k,rows=rows,name=name,**cfg,split=sp,
                      exact=True,replays=64,scales=[1,256],weights=count,tickets_reset=True,
                      shared_bound=shared,registers=regs))
            graphs[name]=graph;objects[name]=probes
        names=list(graphs)
        medians={name:[] for name in names}
        for round_index in range(3):
            samples={name:[] for name in names}
            for iteration in range(20):
                offset=(round_index+iteration)%len(names)
                for name in names[offset:]+names[:offset]:
                    separator.zero_()
                    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record();graphs[name].replay();end.record();end.synchronize()
                    samples[name].append(begin.elapsed_time(end)*1000/count)
            for name in names:medians[name].append(statistics.median(samples[name]))
            emit(dict(test='timing_round',n=n,k=k,rows=rows,round=round_index,samples_us_per_gemm=samples))
        emit(dict(test='timing_summary',n=n,k=k,rows=rows,median_us_per_gemm=medians,
                  production_over_probe={name:[a/b for a,b in zip(medians['baseline'],values)]
                                         for name,values in medians.items() if name!='baseline'}))
        for probes in objects.values():
            for p in probes:p.check()
        del graphs,objects,probes,graph,outputs,refs,base
    del weights
emit(dict(test='complete',decision='DIAGNOSTIC_ONLY',limitation='Isolated compiler liveness and scheduling differ from combined MLP. No speed gate or serving claim.'))
