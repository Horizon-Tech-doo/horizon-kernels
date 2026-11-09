# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Check all MLP tasks, split completion and progress under adversarial CTA orders."""
import ast
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import random
import sys

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('mlp_schedule', root/'mlp_schedule.py')
mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp)
node = next(n for n in ast.parse((root/'runtime.py').read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == 'Schedule')
namespace = {'NCTA_DEFAULT': 188}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'runtime.Schedule', 'exec'), namespace)
Schedule = namespace['Schedule']


def simulate(sc, sg, order):
    cursor = [0] * sc.ncta
    counters = Counter()
    partial = defaultdict(set)
    completed = 0
    total = sum(map(len, sc.lists))
    while completed < total:
        before = completed
        for cta in order:
            if cursor[cta] == len(sc.lists[cta]):
                continue
            task = sc.lists[cta][cursor[cta]]
            if any(c >= 0 and counters[c] < target for c, target in ((task[4], task[5]), (task[6], task[7]))):
                continue
            signal = True
            if task[0] == mlp.GU:
                partial[task[1]].add(task[2])
                signal = len(partial[task[1]]) == sg
            if signal and task[8] >= 0:
                counters[task[8]] += 1
            cursor[cta] += 1
            completed += 1
        if completed == before:
            raise RuntimeError('MLP plan deadlocked')
    assert all(abs(counters[c] - count) < 1e-6 for c, count in sc.targets.items())


cases=0
for hidden,intermediate,sd in ((5120,17408,8),(4096,14336,8),(256,256,2)):
 for rows in (1,4,8,16,32):
  for ncta in (1,17,188,272,376):
   sc=mlp.build_up_fused(Schedule,rows,hidden,intermediate,(1,sd),ncta)
   tiles=intermediate//128;per=tiles//sd
   for cta,expected in enumerate(sc.lists):
    actual=[]
    for bx in range(cta,2*tiles,ncta):
     actual.append((mlp.GU,bx,0,0,-1,0,-1,0,bx) if bx<tiles else
                   (mlp.GU,bx,0,0,bx-tiles,1,-1,0,tiles+(bx-tiles)//per))
    for task in range((cta-2*tiles%ncta+ncta)%ncta,hidden//128*sd,ncta):
     bx,bk=divmod(task,sd);actual.append((mlp.DOWN,bx,bk,0,tiles+bk,per,-1,0,-1))
    assert actual==expected,(rows,ncta,cta)
   shuffled=list(range(ncta));random.Random(5102026).shuffle(shuffled)
   for order in (range(ncta),range(ncta-1,-1,-1),shuffled):simulate(sc,1,order)
   cases+=1
for args in ((4,5120,17408,(2,8),188),(4,5120,17408,(1,8),188,'slice')):
 try:mlp.build_up_fused(Schedule,*args)
 except ValueError:pass
 else:raise AssertionError('unsupported fusion accepted')
print(f'MLP_UP_OWNED_CPU_PASS cases={cases} assignments_exact=True adversarial_progress=True')
