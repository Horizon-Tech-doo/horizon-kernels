# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GPU gate for explicit segment geometry, GU split-K and auxiliary selection.

Arguments: candidate directory, JSONL output. Exclusive lock required.
Synthetic NVFP4 weights; validates arithmetic/scheduling, not model quality/speed.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, silu_mul as sm

candidate=Path(sys.argv[1]); output=open(sys.argv[2],'w')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
sg=load('candidate_segment',candidate/'segment.py')
sg.tk=load('candidate_tasks',candidate/'tasks.py')
nr=load('portable_standalone_norm',candidate/'add_rmsnorm.py')
# Deliberately force GU split-K for generic cases, independently of heuristic
# tuning, to catch the historical missing-split schedule/prefetch assumption.
forced_splits=(4,2,8,4)
sg.bucket=lambda m,nq,geometry: (16 if m<=16 else 32,forced_splits)
torch.manual_seed(290928)
cases=[('legacy',4,2,(5120,17408,6144),16384,-1,True)]
cases += [(f'generic_m{m}_hw{hw}',m,hw,(4096,14336,4096),12288,80,False)
          for m,hw in ((4,0),(4,1),(4,2),(32,0),(32,2))]
cases += [('no_aux_same_nq',4,2,(4096,14336,4096),16384,0,False)]
weights={}
for tag,m,hw,geometry,nq,nba,centered in cases:
    forced_splits=(4,1,8,4) if tag=='legacy' else (4,2,8,4)
    sg.validate_splits(geometry,forced_splits)
    h,i,ko=geometry
    def prep(n,k):
        if (n,k) not in weights:
            weights[n,k]=xl.prepare(*synthetic_layer(n,k,seed=n+k))
        return weights[n,k]
    current={'out':prep(h,ko),'gu':prep(2*i,h),'down':prep(h,i),
             'w2':torch.randn(h,device='cuda',dtype=torch.bfloat16)*0.05}
    nxt={'qkvz':prep(nq,h),'w1':torch.randn(h,device='cuda',dtype=torch.bfloat16)*0.05}
    effective_ba=96 if nba==-1 else nba
    if effective_ba:
        nxt['ba']=torch.randn(effective_ba,h,device='cuda',dtype=torch.bfloat16)*0.01
    seg=sg.Segment(m,geometry=geometry,nq=nq,nba=nba,zero_centered_gamma=centered,
                   hw=hw,ncta=2*torch.cuda.get_device_properties(0).multi_processor_count)
    def gemm(x,prepared,split):
        return xl.linear(x,prepared,cfg=(1,16 if m<=16 else 32,4,1,64,split,4))
    for trial in range(2):
        core=torch.randn(m,ko,device='cuda',dtype=torch.bfloat16)
        residual=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)
        o=gemm(core,current['out'],forced_splits[0])
        y2,r2=nr.add_rmsnorm(o,residual,current['w2'],1e-6,zero_centered_gamma=centered)
        gu=gemm(y2,current['gu'],forced_splits[1]);a=sm.silu_mul(gu,mode=1)
        d=gemm(a,current['down'],forced_splits[2])
        y1,r1=nr.add_rmsnorm(d,r2,nxt['w1'],1e-6,zero_centered_gamma=centered)
        q=gemm(y1,nxt['qkvz'],forced_splits[3])
        ref=dict(o=o,y2=y2,r2=r2,gu=gu,a=a,d=d,y1=y1,r1=r1,q=q)
        if effective_ba:ref['ba']=sk.linear(y1,nxt['ba'])
        q_out=torch.empty_like(q);r_out=torch.empty_like(r1)
        ba_out=torch.empty(m,seg.nba,device='cuda',dtype=torch.bfloat16)
        def run():seg(core,residual,current,nxt,q=q_out,ba=ba_out,r1=r_out)
        def check():
            torch.cuda.synchronize()
            assert not seg.failed(),(tag,'kernel timeout')
            got=dict(seg.ws,q=q_out,r1=r_out,ba=ba_out)
            for name,value in ref.items():
                assert torch.isfinite(got[name]).all(),(tag,trial,name,'nonfinite')
                assert torch.equal(got[name],value),(tag,trial,name,'not exact')
            assert not torch.count_nonzero(seg.tickets).item()
            assert not torch.count_nonzero(seg.ts['counters']).item()
            assert not torch.count_nonzero(seg.ts['done']).item()
        run();check();run();check()
        for attempt in range(2):
            graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph,capture_error_mode='thread_local'):run()
                break
            except RuntimeError:
                if attempt:raise
                torch.cuda.synchronize()
        for _ in range(2):
            for value in (*seg.ws.values(),q_out,r_out,ba_out):value.fill_(float('nan'))
            graph.replay();check()
            if not effective_ba:assert torch.isnan(ba_out).all(),(tag,'unexpected auxiliary write')
    row=dict(tag=tag,m=m,hw=hw,geometry=geometry,nq=nq,nba=nba,splits=forced_splits,
             exact=True,graph=True,counters_reset=True,trials=2)
    line=json.dumps(row);print(line,flush=True);output.write(line+'\n');output.flush()
    del graph,seg
output.write(json.dumps({'complete':True,'cases':len(cases)})+'\n');output.close()
print('SEGMENT_GEOMETRY_GPU_PASS',len(cases),flush=True)
