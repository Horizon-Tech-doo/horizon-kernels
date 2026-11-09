# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the factory's actual output-coordinate expressions and prove unique coverage."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

root=Path(sys.argv[1])
factory=next(n for n in ast.parse((root/'tasks.py').read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name=='gemm_task_hw')
body=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='body')
expressions={}
for node in ast.walk(body):
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in ('warp','lane','wn','warp_m','grow','gcol'):expressions[name]=compile(ast.Expression(node.value),'coordinate','eval')
assert set(expressions)=={'warp','lane','wn','warp_m','grow','gcol'}
emitter_source=root.parents[1]/'cuda/intrinsics/macro/mma_macro_generator.py'
emitter_tree=ast.parse(emitter_source.read_text())
emitter_class=next(n for n in emitter_tree.body if isinstance(n,ast.ClassDef) and n.name=='TensorCoreIntrinEmitter')
init=next(n for n in emitter_class.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
defaults=dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):],init.args.defaults))
assert ast.literal_eval(defaults['is_m_first']) is False
binding=next(n for n in emitter_class.body if isinstance(n,ast.FunctionDef) and n.name=='extract_thread_binding')
for arg in binding.args.args:arg.annotation=None
binding.returns=None
binding_ns={};exec(compile(ast.Module(body=[binding],type_ignores=[]),'emitter','exec'),binding_ns)
call=next(n for n in factory.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call)
          and isinstance(n.value.func,ast.Name) and n.value.func.id=='TensorCoreIntrinEmitter').value
assert all(k.arg!='is_m_first' for k in call.keywords)
assert any(k.arg=='block_row_warps' and isinstance(k.value,ast.Name) and k.value.id=='row_warps' for k in call.keywords)
assert any(k.arg=='warp_row_tiles' and isinstance(k.value,ast.Name) and k.value.id=='ROW_WM' for k in call.keywords)
points=0
for wm,row_warps,width in ((16,1,128),(16,1,256),(32,1,128),(32,1,256),(32,2,128),(64,2,128)):
    seen=set()
    for tid in range(width*row_warps):
        ns=dict(tid=tid,row_warps=row_warps,ROW_WM=wm//row_warps,BLOCK_N=width,NB=1,bx=3)
        for name in ('warp','lane','wn','warp_m'):ns[name]=eval(expressions[name],{},ns)
        emitter=SimpleNamespace(WARP_SIZE=32,block_row_warps=row_warps,block_col_warps=width//32,is_m_first=False)
        assert binding_ns['extract_thread_binding'](emitter,tid)==(ns['lane'],ns['wn'],ns['warp_m'])
        for i in range(wm//row_warps//16):
            for j in range(2):
                for r2 in range(4):
                    ns.update(i=i,b=0,j=j,r2=r2)
                    row=eval(expressions['grow'],{},ns)
                    column=eval(expressions['gcol'],{},ns)
                    for v in range(2):
                        point=(row,column+v)
                        assert point not in seen;seen.add(point)
    assert seen=={(r,3*width+c) for r in range(wm) for c in range(width)}
    points+=len(seen)
print('ROW_WARP_LAYOUT_CPU_PASS',points,'unique coordinates in six geometries; actual emitter binding checked')
