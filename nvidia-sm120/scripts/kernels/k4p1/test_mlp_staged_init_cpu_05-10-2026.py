# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Actual source placement and worst-case copy-group completion proof."""
import ast
from collections import deque
from pathlib import Path
import sys

tree=ast.parse(Path(sys.argv[1]).read_text())
factory=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gemm_task_hw')
body=next(n for n in factory.body if isinstance(n,ast.FunctionDef) and n.name=='body')
initial=[n for n in body.body if isinstance(n,ast.For) and ast.unparse(n.target)=='p']
assert len(initial)==1
loop=initial[0]
assert ast.unparse(loop.iter)=='T.unroll(S - 1)'
assert len(loop.body)==2 and isinstance(loop.body[1],ast.If)
assert ast.unparse(loop.body[1].test)=='staged_init'
assert '"xl_commit"' in ast.unparse(loop.body[1]).replace("'",'"')
serial=next(n for n in body.body if isinstance(n,ast.For) and ast.unparse(n.target)=='kt')
condition=ast.parse('kt == 0 and not staged_init',mode='eval').body
assert ast.dump(serial.body[0].test)==ast.dump(condition)
assert 'xl_wait<0>' in ast.unparse(serial.body[0].body)
assert 'S - 2' in ast.unparse(serial.body[0].orelse)
cases=0;peak=0
for stages in (3,4):
    for count in (stages-1,stages,stages+1,20,40,68,80,136,272):
        for chunks in (2,4):
            for staged in (False,True):
                for late in (False,True):
                    slots={'W':[None]*stages,'A':[None]*stages};pending=deque()
                    for index in range(stages-1):pending.append([('W',index,index)])
                    if staged:
                        for index in range(stages-1):pending.append([('A',index,index)])
                    else:pending.append([('A',index,index) for index in range(stages-1)])
                    peak=max(peak,len(pending))
                    def wait(remaining):
                        while len(pending)>remaining:
                            for kind,index,value in pending.popleft():slots[kind][index]=value
                    for kt in range(count):
                        wait(0 if kt==0 and not staged else stages-2)
                        current=kt%stages;destination=(kt+stages-1)%stages
                        assert current!=destination
                        def issue():
                            next_tile=kt+stages-1
                            pending.append([(kind,destination,next_tile) for kind in ('W','A')]
                                           if next_tile<count else [])
                        if not late:issue()
                        for c in range(chunks):
                            for half in range(2):
                                assert slots['W'][current]==slots['A'][current]==kt
                                if late and c==half==0:issue()
                        assert len(pending)<=stages-1
                    wait(0);assert not pending;cases+=1
assert peak==6
print(f'STAGED_INIT_RING_PROOF_PASS cases={cases} peak_initial_groups={peak}; GPU checks pending')
