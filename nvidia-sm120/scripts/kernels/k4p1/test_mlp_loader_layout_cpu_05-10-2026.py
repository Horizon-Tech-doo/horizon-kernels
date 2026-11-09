# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compare handwritten activation-copy addresses to TileLang's layout function."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
root=Path(sys.argv[1])
layout=Path(sys.argv[2]) if len(sys.argv)>2 else root.parents[1]/'cuda/intrinsics/layout/mma_layout.py'
node=next(n for n in ast.parse(layout.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_swizzle_layout')
for arg in node.args.args:arg.annotation=None
node.returns=None
ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),'layout','exec'),ns)
tasks=ast.parse((root/'tasks.py').read_text())
fn=next(n for n in tasks.body if isinstance(n,ast.FunctionDef) and n.name=='gemm_task_hw')
call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_hw_loader_src')
assert len(call.args) in (4,5,6) and isinstance(call.args[3],ast.Name) and call.args[3].id=='block_k'
if len(call.args)>=5:
 assert isinstance(call.args[4],ast.Name) and call.args[4].id=='block_n'
if len(call.args)==6:
 assert isinstance(call.args[5],ast.Name) and call.args[5].id=='THREADS'
points=0
for wm in (16,32):
 for bk in (64,128):
  for stages in (3,4,5,6):
   seen=set()
   for st in range(stages):
    for row in range(wm):
     for col in range(bk):
      tile,r,c=ns['get_swizzle_layout'](row,col,bk,SimpleNamespace(bits=16))
      expected=((st*(bk//64)+tile)*wm+r)*128+c*2
      ch,v=divmod(col,8)
      actual=((st*(bk//64)+ch//8)*wm+row)*128+(((ch&7)^(row&7))*16)+v*2
      assert actual==expected,(wm,bk,st,row,col)
      assert actual not in seen;seen.add(actual);points+=1
   assert len(seen)==stages*wm*bk and max(seen)==stages*wm*bk*2-2
print(f'MLP_LOADER_LAYOUT_CPU_PASS points={points} bijection=True forwarding_checked=True')
