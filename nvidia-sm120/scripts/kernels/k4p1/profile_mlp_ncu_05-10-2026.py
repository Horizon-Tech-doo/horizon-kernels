# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Nsight target: synthetic four-layer graph; checks exactness before/after profiling.

Application replay uses per-process evidence files. Profiler timings are
 diagnostic only; use the paired uninstrumented benchmark for speed claims.
"""
import json
import os
from pathlib import Path
import sys
import torch
import tilelang.contrib
root, output = map(Path,sys.argv[1:3]);rows=int(sys.argv[3]);kind=sys.argv[4]
assert rows in (4,32) and kind in ('baseline','k64','k128')
tilelang.contrib.__path__.insert(0,str(root/'contrib'))
from tilelang.contrib.horizon_k4.mlp_static_plan import StaticMlp
from tilelang.contrib.horizon_k4.mlp_up_fusion import UpFusionMlp
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl, decode_regfrag as rf
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
def split(n, k, rows):
    cfg = xl.config(n, k, rows)
    return (tuple(cfg[1:]) if len(cfg) > 1 else rf.config(n, k, rows))[3] if cfg[0] == 'rf' else cfg[5]

def capture(fn):
    for attempt in range(4):
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                result = fn()
            return graph, result
        except RuntimeError:
            torch.cuda.synchronize()
            if attempt == 3:
                raise

props=torch.cuda.get_device_properties(0)
hidden,intermediate=5120,17408
layers=[]
for layer in range(4):
 prepared=[]
 for j,(n,k) in enumerate(((2*intermediate,hidden),(hidden,intermediate))):
  raw=synthetic_layer(n,k,seed=1052026+10*layer+j);prepared.append(xl.prepare(*raw));del raw
 layers.append(tuple(prepared))
weight_bytes=sum(t.numel()*t.element_size() for pair in layers for prep in pair for t in (prep.Wg,prep.Sg,prep.alpha))
assert weight_bytes>512*1024**2
separator=torch.empty(256*1024**2,device='cuda',dtype=torch.uint8)
activation=torch.compile(lambda z:torch.nn.functional.silu(z[:,:intermediate])*z[:,intermediate:],fullgraph=True,dynamic=False)
torch.manual_seed(1052026+rows);x=torch.randn(rows,hidden,device='cuda',dtype=torch.bfloat16)
def baseline():
 value=x
 for gate,down in layers:value=xl.linear(activation(xl.linear(value,gate)),down)
 return value
baseline();torch.cuda.synchronize()
sg,sd=split(2*intermediate,hidden,rows),split(hidden,intermediate,rows)
blocks=[];grid=None;shared_bound=None
if kind!='baseline':
 bk,stages=(128,3) if kind=='k128' else (64,4)
 wm=16 if rows<=16 else 32
 shared_bound=stages*(wm*bk*2+4*(bk//32)*576)+1024
 resident=min(2,props.shared_memory_per_multiprocessor//shared_bound)
 assert resident>=1;grid=resident*props.multi_processor_count
 cls=UpFusionMlp if kind=='k128' else StaticMlp
 blocks=[cls(rows,hidden,intermediate,(sg,sd),grid,stages=stages,block_k=bk) for _ in layers]
def candidate():
 value=x
 for block,pair in zip(blocks,layers):value=block(value,*pair)
 return value
if blocks:
 for scale in (1.,256.):
  x.normal_();x.mul_(scale);value=x;reference=x
  for block,(gate,down) in zip(blocks,layers):
   gu=xl.linear(reference,gate);a=activation(gu);reference=xl.linear(a,down)
   value=block(value,gate,down);torch.cuda.synchronize();block.check()
   assert all(torch.equal(t,r) for t,r in ((block.gu,gu),(block.a,a),(value,reference)))
x.normal_();expected=baseline().clone();torch.cuda.synchronize()
graph,result=capture(candidate if blocks else baseline)
for _ in range(2):
 graph.replay();torch.cuda.synchronize();assert torch.equal(result,expected)
 for block in blocks:block.check()
separator.zero_();torch.cuda.synchronize()
torch.cuda.nvtx.range_push('hz_profile')
graph.replay();torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()
assert torch.equal(result,expected)
for block in blocks:block.check()
evidence=output.with_name(output.stem+f'-pid{os.getpid()}'+output.suffix)
evidence.write_text(json.dumps(dict(rows=rows,kind=kind,weight_bytes=weight_bytes,graph_layers=4,
 grid=grid,shared_bound=shared_bound,exact=True,counters_reset=True,pid=os.getpid()),indent=2)+'\n')
print('NCU_TARGET_COMPLETE',evidence,flush=True)
