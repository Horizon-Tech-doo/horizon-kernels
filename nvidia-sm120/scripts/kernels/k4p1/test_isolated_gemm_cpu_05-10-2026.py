# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Assignment coverage for the actual isolated-task arithmetic."""
import ast,json,sys
from collections import Counter
from pathlib import Path
from isolated_gemm_configs_05_10_helper import configs
root=Path(sys.argv[1])
text=ast.unparse(ast.parse((root/'gemm_isolated_probe.py').read_text()))
assert 'task = cta + wave * grid' in text and 'bx, bk = (task // split, task % split)' in text
assert 'gemm_task_hw' in text and 'hz_wait' not in text
cases=0
for n,k in ((34816,5120),(5120,17408)):
 for rows in (4,32):
  sp=1 if n==34816 else (8 if rows==4 else 4)
  for cfg in configs(rows,n):
   grid=cfg['grid'];width=cfg['width'];total=n//width*sp
   observed=Counter((task//sp,task%sp) for cta in range(grid)
                    for wave in range((total+grid-1)//grid)
                    if (task:=cta+wave*grid)<total)
   assert observed==Counter({(bx,bk):1 for bx in range(n//width) for bk in range(sp)})
   assert k%(sp*64)==0
   cases+=1
assert cases==20
print(json.dumps(dict(decision='PASS',cases=cases,actual_source_assignment_checked=True,
                     all_tasks_once=True,no_producer_wait=True)))
