# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exercise progress of each CTA's task queue under three adversarial visit orders."""
import importlib.util
from pathlib import Path
import random
import sys
s=importlib.util.spec_from_file_location('wide',Path(sys.argv[1])/'mlp_wide_plan.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
n=0
for h,i in ((5120,17408),(4096,14336),(2048,8192)):
 for sd in (1,2,4,8):
  for width in (128,256):
   for ct in (188,376):
    plan=m.WidePlan(h,i,(1,sd),ct,width)
    for mode in (0,1,2):
     counts=[0]*plan.n_counters;pos=[0]*ct;order=list(range(ct))
     if mode==1:order.reverse()
     if mode==2:random.Random(1052026).shuffle(order)
     while sum(pos)<sum(map(len,plan.lists)):
      progress=False
      for c in order:
       if pos[c]==len(plan.lists[c]):continue
       _,_,_,waits,signals=plan.lists[c][pos[c]]
       if all(counts[k]>=v for k,v in waits):
        for k in signals:counts[k]+=1
        pos[c]+=1;progress=True
      assert progress,'protocol deadlock'
    n+=1
for args in ((5120,17408,(2,8),188,256),(5120,17408,(1,8),188,192)):
 try:m.WidePlan(*args)
 except ValueError:pass
 else:raise AssertionError('unsupported geometry accepted')
print('WIDE_PROTOCOL_CPU_PASS',n,'geometries; forward/reverse/shuffled; invalid geometry rejected')
