# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Proof of packed reduction alignment, coverage, bounds and split order."""
import ast,json,sys
from pathlib import Path
root=Path(sys.argv[1]);tree=ast.parse((root/'tasks.py').read_text())
factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gemm_task_hw')
branch=next(n for n in ast.walk(factory) if isinstance(n,ast.If) and ast.unparse(n.test)=='partial_vector8')
text=ast.unparse(ast.Module(body=branch.body,type_ignores=[]))
assert 'ld.global.cg.v4.u64' in (root/'tasks.py').read_text()
assert 'octet = tid + batch * THREADS' in text
assert 'row = octet // (BLOCK_N // 8)' in text
assert 'col = bx * BLOCK_N + octet % (BLOCK_N // 8) * 8' in text
assert 'for sk in T.unroll(1, split_k)' in text
assert 'acc8[v] = acc8[v] + piece8[v]' in text
assert 'for v in T.unroll(8)' in text and 'for v in T.vectorized(8)' in text
assert 'T.access_ptr(P[0, row, col]' in text and 'T.access_ptr(P[sk, row, col]' in text
assignments={n.targets[0].id:compile(ast.Expression(n.value),'actual-assignment','eval')
             for n in ast.walk(ast.Module(body=branch.body,type_ignores=[])) if isinstance(n,ast.Assign) and len(n.targets)==1
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('octet','row','col')}
assert set(assignments)=={'octet','row','col'}
cases=loads=0
for rows in (1,4,8,16,32):
 wm=16 if rows<=16 else 32
 for width,threads in ((128,128),(128,256),(256,256)):
  for split in (4,8):
   n=5120;seen=set()
   for bx in range(n//width):
    for batch in range(wm*(width//8)//threads):
     for tid in range(threads):
      env={'tid':tid,'batch':batch,'THREADS':threads,'BLOCK_N':width,'bx':bx}
      for name in ('octet','row','col'):env[name]=eval(assignments[name],{'__builtins__':{}},env)
      row,col=env['row'],env['col']
      if row>=rows:continue
      for sk in range(split):
       address=((sk*rows+row)*n+col)*4
       assert address%32==0 and 0<=address and address+32<=split*rows*n*4
       assert col//width==bx and (col+7)//width==bx
       loads+=1
      for v in range(8):
       assert (row,col+v) not in seen;seen.add((row,col+v))
   assert seen=={(r,c) for r in range(rows) for c in range(n)}
   cases+=1
print(json.dumps(dict(decision='PASS',cases=cases,packed_loads_checked=loads,
                     actual_assignment_checked=True,aligned_32_bytes=True,
                     output_coverage_once=True,first_split_then_ascending_additions=True)))

source=(root/'tasks.py').read_text()
for v,reg in enumerate('aabbccdd'):
 expression=f'static_cast<unsigned>({reg}' + (' >> 32)' if v%2 else ')')
 assert f'out[{v}]=__uint_as_float({expression});' in source
assert 'type(partial_vector8) is not bool or partial_vector4 and partial_vector8' in source
mixed=(root/'mlp_mixed.py').read_text()
assert mixed.count('down_partial_vector8=False):')==2
assert 'partial_vector8=down_partial_vector8' in mixed
print('VECTOR8_BIT_REINTERPRETATION_AND_DEFAULT_FORWARDING_PASS')
