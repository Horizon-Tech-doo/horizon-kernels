# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""CPU task conservation and forward-progress check; no GPU imports."""
import ast
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import sys
from grouped_gu_schedule import schedule_type
segment,runtime=map(Path,sys.argv[1:3])
def extract(path,names,ns):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns
base=extract(runtime,{'Schedule'},{'NCTA_DEFAULT':188})['Schedule']
ns=dict(H=5120,I=17408,KO=6144,NQ=16384,NBA=96,DEFAULT_GEOMETRY=(5120,17408,6144),DEFAULT_GDN_HEADS=(16,48),tk=SimpleNamespace(BLOCK_N=128,BLOCK_K=64))
ns.update(zip(('OP_OUT','OP_N2','OP_GU','OP_SILU','OP_DOWN','OP_N1','OP_QKVZ','OP_BA','OP_CONV','OP_CORE','OP_GNORM'),range(1,12)))
extract(segment,{'segment_geometry','validate_splits','build_schedule'},ns)
cases=0
for geometry,nq,splits in (((5120,17408,6144),16384,(4,1,8,4)),((4096,14336,4096),12288,(4,2,8,4))):
 for m in (1,4,16,32):
  for ncta in (1,32,188,376):
   schedules=[]
   for cls in (base,schedule_type(base,3,geometry[1],128,splits[2])):
    ns['rt']=SimpleNamespace(Schedule=cls)
    sc=ns['build_schedule'](m,ncta,splits,(4,1,128),nq,geometry=geometry,nba=80)
    sc.check();schedules.append(sc)
   a,b=schedules
   assert Counter(d for q in a.lists for d in q)==Counter(d for q in b.lists for d in q)
   assert a.ops==b.ops and a.targets==b.targets
   for order in (range(ncta),range(ncta-1,-1,-1)):
    cursor=[0]*ncta;counts=Counter();partial=defaultdict(set)
    sp=dict(zip((1,3,5,7),splits));done=0;target=sum(map(len,b.lists))
    while done<target:
     before=done
     for c in order:
      if cursor[c]==len(b.lists[c]):continue
      d=b.lists[c][cursor[c]]
      if any(w>=0 and counts[w]<t for w,t in ((d[4],d[5]),(d[6],d[7]))):continue
      op,bx,bk=d[:3]
      if op in sp:
       partial[op,bx].add(bk);signal=len(partial[op,bx])==sp[op]
      else:signal=True
      if signal and d[8]>=0:counts[d[8]]+=1
      cursor[c]+=1;done+=1
     assert done>before,'schedule deadlock'
    assert all(abs(counts[c]-v)<1e-6 for c,v in b.targets.items())
   cases+=1
print('GROUPED_GU_CPU_PASS',cases)
