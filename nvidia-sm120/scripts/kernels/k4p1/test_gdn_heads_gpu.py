# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Fused GDN geometry vs standalone conv/GDN plus segment, including state/replay.

Args: candidate directory, JSONL output. Exclusive GPU lock required. Synthetic
weights; tests fusion and layout, not a new model's quality or speed.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from tilelang.contrib.horizon_ops import conv1d_update as cv, gdn_decode as gd
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer

root=Path(sys.argv[1]);out=open(sys.argv[2],'w')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sg=load('heads_segment',root/'segment.py');sg.tk=load('heads_tasks',root/'tasks.py')
weights={};count=0
torch.manual_seed(290928)
for heads,geometry,hw in (((16,48),(5120,17408,6144),2),((8,32),(4096,14336,4096),0),((8,32),(4096,14336,4096),2)):
    hk,hv=heads;h,i,ko=geometry
    _,_,gc,gnq=sg.gdn_geometry(heads,ko)
    nq=14336 if h==5120 else 12288
    splits=(6,1,8,4) if h==5120 else (4,2,8,4)
    sg.bucket=lambda m,nq,geometry:(16,splits)
    def prep(n,k):
        if (n,k) not in weights:weights[n,k]=xl.prepare(*synthetic_layer(n,k,seed=n+k))
        return weights[n,k]
    current=dict(out=prep(h,ko),gu=prep(2*i,h),down=prep(h,i),w2=torch.randn(h,device='cuda',dtype=torch.bfloat16)*.05)
    nxt=dict(qkvz=prep(nq,h),w1=torch.randn(h,device='cuda',dtype=torch.bfloat16)*.05)
    for lengths,m in (((4,),4),((1,3),8)):
        r=len(lengths);slots=r*4+1
        st=dict(conv_state=torch.randn(slots,gc,6,device='cuda',dtype=torch.bfloat16)*.1,
                conv_w=torch.randn(gc,4,device='cuda',dtype=torch.bfloat16)*.1,
                ssm=torch.randn(slots,hv,128,128,device='cuda')*.01,
                sidx=torch.arange(1,slots,device='cuda',dtype=torch.int32).view(r,4),
                cu=torch.tensor([0]+[sum(lengths[:j+1]) for j in range(r)],device='cuda',dtype=torch.int32),
                nacc=torch.ones(r,device='cuda',dtype=torch.int32),
                a_log=torch.randn(hv,device='cuda')*.1,dt_bias=torch.randn(hv,device='cuda')*.1,
                norm_w=torch.ones(128,device='cuda'),lmax=4,scale=128**-.5,eps=1e-6)
        qkvz=torch.randn(m,gnq,device='cuda',dtype=torch.bfloat16)
        ba=torch.randn(m,2*hv,device='cuda',dtype=torch.bfloat16)
        res=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)
        common=dict(geometry=geometry,nq=nq,nba=0,zero_centered_gamma=(h==5120),hw=hw,
                    ncta=2*torch.cuda.get_device_properties(0).multi_processor_count)
        fused=sg.GdnLayer(m,r,st,gdn_heads=heads,**common)
        chain=sg.Segment(m,**common)
        sr={k:(v.clone() if torch.is_tensor(v) else v) for k,v in st.items()}
        sf={k:(v.clone() if torch.is_tensor(v) else v) for k,v in st.items()}
        conv=torch.zeros(m,gc,device='cuda',dtype=torch.bfloat16)
        cv.conv_update(qkvz[:,:gc],sr['conv_state'],sr['conv_w'],sr['sidx'],sr['nacc'],sr['cu'],4,out=conv)
        z=qkvz[:,gc:].unflatten(1,(hv,128))
        core=gd.fused_gdn_decode_post_conv_mtp(conv,ba[:,hv:],ba[:,:hv],sr['a_log'],sr['dt_bias'],sr['sidx'],sr['cu'],sr['nacc'],sr['ssm'],z,sr['norm_w'],out=torch.zeros(m,hv,128,device='cuda',dtype=torch.bfloat16),scale=sr['scale'],norm_eps=sr['eps']).flatten(1)
        qr,_,rr=chain(core,res,current,nxt)
        q=torch.empty_like(qr);ro=torch.empty_like(rr);bo=torch.empty(m,1,device='cuda',dtype=torch.bfloat16)
        def reset():
            for key in ('conv_state','ssm'):sf[key].copy_(st[key])
            fused.core.fill_(float('nan'));fused.conv_out.fill_(float('nan'))
        def run():fused(qkvz,ba,res,sf,current,nxt,q=q,ba=bo,r1=ro)
        def check():
            torch.cuda.synchronize()
            assert not fused.failed()
            assert torch.equal(q,qr) and torch.equal(ro,rr),(heads,hw,lengths,'outputs')
            assert torch.equal(fused.core,core),(heads,hw,lengths,'core/tail')
            assert torch.equal(fused.conv_out[:sum(lengths)],conv[:sum(lengths)]),(heads,hw,lengths,'conv')
            for key in ('conv_state','ssm'):assert torch.equal(sf[key],sr[key]),(heads,hw,lengths,key)
            for key in fused.ws:assert torch.equal(fused.ws[key],chain.ws[key]),(heads,hw,lengths,key)
            assert not torch.count_nonzero(fused.tickets).item()
            assert not torch.count_nonzero(fused.ts['counters']).item()
            assert not torch.count_nonzero(fused.ts['done']).item()
        reset();run();check();reset();run();check()
        for attempt in range(2):
            reset();graph=torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph,capture_error_mode='thread_local'):run()
                break
            except RuntimeError:
                if attempt:raise
                torch.cuda.synchronize()
        for _ in range(2):reset();graph.replay();check()
        row=dict(heads=heads,geometry=geometry,hw=hw,lengths=lengths,m=m,exact=True,state_exact=True,graph=True,padding_checked=sum(lengths)<m)
        text=json.dumps(row);print(text,flush=True);out.write(text+'\n');out.flush();count+=1
        del graph,fused,chain
out.write(json.dumps(dict(complete=True,cases=count))+'\n');out.close()
