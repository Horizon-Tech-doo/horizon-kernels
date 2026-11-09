# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Worst-case async completion/ring proof for the actual late-issue placement."""
import ast
from collections import deque
from pathlib import Path
import sys

source=Path(sys.argv[1]).read_text()
tree=ast.parse(source)
factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gemm_task_hw')
body=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='body')
condition=ast.dump(ast.parse('late_issue and c == 0 and s == 0',mode='eval').body)
late=[n for n in ast.walk(body) if isinstance(n,ast.If) and ast.dump(n.test)==condition]
assert len(late)==1
parents={child:node for node in ast.walk(body) for child in ast.iter_child_nodes(node)}
sloop=parents[late[0]]
assert isinstance(sloop,ast.For) and ast.unparse(sloop.target)=='s'
assert ast.unparse(sloop.iter)=='T.unroll(2)'
assert isinstance(sloop.body[0],ast.If) and 'COMPUTE_THREADS' in ast.unparse(sloop.body[0].test)
assert any(isinstance(n,ast.Call) and ast.unparse(n.func)=='T.ptx_mma' for n in ast.walk(sloop.body[0]))
assert sloop.body[-1] is late[0]  # all threads issue, including non-compute threads
assert len(late[0].body)==1 and ast.unparse(late[0].body[0]).startswith('issue(')
cases=0
for stages in (3,4,6):
    for kchunks in (2,4):
        for count in (stages-1,stages,stages+1,20,40,68,80,136,272):
            for delayed in (False,True):
                slots=list(range(stages-1))+[None]
                pending=deque()
                for kt in range(count):
                    while len(pending)>stages-2:
                        index,value=pending.popleft()
                        if value is not None:slots[index]=value
                    current=kt%stages;destination=(kt+stages-1)%stages
                    assert slots[current]==kt
                    assert destination!=current
                    issued=0
                    for c in range(kchunks):
                        for half in range(2):
                            if not delayed and c==half==0:
                                pending.append((destination,kt+stages-1 if kt+stages-1<count else None));issued+=1
                            assert slots[current]==kt  # each MMA reads the same ascending K chunk
                            if delayed and c==half==0:
                                pending.append((destination,kt+stages-1 if kt+stages-1<count else None));issued+=1
                    assert issued==1
                while pending:pending.popleft()
                cases+=1
print(f'LATE_ISSUE_RING_PROOF_PASS cases={cases}; GPU identity and speed pending')
