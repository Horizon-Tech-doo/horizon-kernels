# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Verify arithmetic per-CTA assignments equal the independent descriptor scheduler."""
import ast
import importlib.util
from pathlib import Path
import sys
root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('schedule', root/'mlp_schedule.py')
mlp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mlp)
node = next(n for n in ast.parse((root/'runtime.py').read_text()).body
            if isinstance(n,ast.ClassDef) and n.name=='Schedule')
ns={'NCTA_DEFAULT':188};exec(compile(ast.Module(body=[node],type_ignores=[]),'Schedule','exec'),ns)
cases=0
for hidden,intermediate,sg,sd in ((5120,17408,1,8),(4096,14336,2,8),(256,256,1,2)):
 for rows in (1,4,8,16,32):
  for ncta in (1,17,188,272,376):
   schedule=mlp.build(ns['Schedule'],rows,hidden,intermediate,(sg,sd),ncta)
   tiles=intermediate//128;per=tiles//sd;cols=intermediate//sd
   ngu=2*tiles*sg;nact=rows*sd;ndown=hidden//128*sd
   for cta,expected in enumerate(schedule.lists):
    actual=[]
    for task in range(cta,ngu,ncta):
     bx,bk=divmod(task,sg);actual.append((mlp.GU,bx,bk,0,-1,0,-1,0,(bx%tiles)//per))
    for task in range((cta-ngu%ncta+ncta)%ncta,nact,ncta):
     sl,row=divmod(task,rows);actual.append((mlp.SILU,row,sl*cols,0,sl,2*per,-1,0,sd+sl))
    for task in range((cta-(ngu+nact)%ncta+ncta)%ncta,ndown,ncta):
     bx,bk=divmod(task,sd);actual.append((mlp.DOWN,bx,bk,0,sd+bk,rows,-1,0,-1))
    assert actual==expected,(rows,ncta,cta)
   cases+=1
print(f'MLP_ARITHMETIC_PLAN_CPU_PASS cases={cases} all_cta_assignments_exact=True')
