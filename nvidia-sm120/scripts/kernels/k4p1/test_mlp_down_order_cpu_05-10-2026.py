# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Actual-source DOWN assignment, task coverage and adversarial progress."""
import ast,importlib.util,json,random,sys
from collections import Counter
from pathlib import Path
root=Path(sys.argv[1]);tree=ast.parse((root/'mlp_mixed.py').read_text());text=ast.unparse(tree)
assert 'bx, bk = (task % (hidden // block_n_down), task // (hidden // block_n_down))' in text
assert 'bx, bk = (task // sd, task % sd)' in text
assert 'if down_split_major:' in text
spec=importlib.util.spec_from_file_location('wide',root/'mlp_wide_plan.py');wide=importlib.util.module_from_spec(spec);spec.loader.exec_module(wide)
cases=0
for rows in (1,4,8,16,32):
 for sd in (4,8):
  for grid in (272,376):
   controls=[]
   for split_major in (False,True):
    plan=wide.WidePlan(5120,17408,(1,sd),grid,128,128,down_split_major=split_major)
    tiles=136;total=40*sd
    actual=[[] for _ in range(grid)]
    for cta in range(grid):
     for t in range(cta,272,grid):
      actual[cta].append(('GU',t,0,[] if t<tiles else [(t-tiles,1)],
                          [t] if t<tiles else [tiles+(t-tiles)//(tiles//sd)]))
     for t in range((cta-272%grid+grid)%grid,total,grid):
      bx,bk=(t%40,t//40) if split_major else divmod(t,sd)
      actual[cta].append(('DOWN',bx,bk,[(tiles+bk,tiles//sd)],[]))
    assert actual==plan.lists
    controls.append(Counter(json.dumps(t) for lst in plan.lists for t in lst))
    shuffled=list(range(grid));random.Random(5102026).shuffle(shuffled)
    for order in (range(grid),range(grid-1,-1,-1),shuffled):
     cursor=[0]*grid;counts=Counter();done=0;expected=272+total
     while done<expected:
      before=done
      for cta in order:
       if cursor[cta]==len(plan.lists[cta]):continue
       _,_,_,waits,signals=plan.lists[cta][cursor[cta]]
       if any(counts[c]<target for c,target in waits):continue
       for c in signals:counts[c]+=1
       cursor[cta]+=1;done+=1
      assert done>before,'deadlocked'
     assert [counts[c] for c in range(tiles)]==[1]*tiles
     assert [counts[c] for c in range(tiles,tiles+sd)]==[tiles//sd]*sd
    cases+=1
   assert controls[0]==controls[1],'task multiset differs'
assert cases==40
print(json.dumps(dict(decision='PASS',cases=cases,actual_assignment_checked=True,
                     unchanged_task_multiset=True,adversarial_progress=True)))
