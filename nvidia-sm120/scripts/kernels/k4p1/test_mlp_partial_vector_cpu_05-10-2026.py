# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Proof of packed reduction alignment, coverage, bounds and split order."""
import ast,json,sys
from pathlib import Path
root=Path(sys.argv[1]);tree=ast.parse((root/'tasks.py').read_text())
factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gemm_task_hw')
branch=next(n for n in ast.walk(factory) if isinstance(n,ast.If) and ast.unparse(n.test)=='partial_vector4')
text=ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
assert 'ld.global.cg.v4.f32' in (root/'tasks.py').read_text()
assert 'quad = tid + batch * THREADS' in text
assert 'row = quad // (BLOCK_N // 4)' in text
assert 'col = bx * BLOCK_N + quad % (BLOCK_N // 4) * 4' in text
assert 'for sk in T.unroll(1, split_k)' in text
assert 'acc4[v] = acc4[v] + piece4[v]' in text
assert 'T.access_ptr(P[0, row, col]' in text and 'T.access_ptr(P[sk, row, col]' in text
cases=loads=0
for rows in (1,4,8,16,32):
 wm=16 if rows<=16 else 32
 for width,threads in ((128,128),(128,256),(256,256)):
  for split in (4,8):
   n=5120;seen=set()
   for bx in range(n//width):
    for batch in range(wm*(width//4)//threads):
     for tid in range(threads):
      quad=tid+batch*threads;row,col=quad//(width//4),bx*width+quad%(width//4)*4
      if row>=rows:continue
      for sk in range(split):
       address=((sk*rows+row)*n+col)*4
       assert address%16==0 and 0<=address and address+16<=split*rows*n*4
       assert col//width==bx and (col+3)//width==bx
       loads+=1
      for v in range(4):
       assert (row,col+v) not in seen;seen.add((row,col+v))
   assert seen=={(r,c) for r in range(rows) for c in range(n)}
   cases+=1
print(json.dumps(dict(decision='PASS',cases=cases,packed_loads_checked=loads,
                     actual_assignment_checked=True,aligned_16_bytes=True,
                     output_coverage_once=True,first_split_then_ascending_additions=True)))
