# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exact independent alignment oracle, repeated launches and graph replays."""
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0, sys.argv[1])
from stable_align import align


def reference(values, experts, bs):
    # Build separate lists; deliberately do not use the kernel's histogram/rank algorithm.
    size = len(values) + experts * (bs - 1)
    if len(values) < experts:
        size = min(len(values) * bs, size)
    groups = [[i for i, e in enumerate(values) if e == expert] for expert in range(experts)]
    sorted_ids, expert_ids = [], []
    for expert, group in enumerate(groups):
        padded = ((len(group) + bs - 1) // bs) * bs
        sorted_ids.extend(group + [len(values)] * (padded - len(group)))
        expert_ids.extend([expert] * (padded // bs))
    used = len(sorted_ids)
    sorted_ids.extend([len(values)] * (size - used))
    expert_ids.extend([-1] * ((size + bs - 1) // bs - len(expert_ids)))
    return [sorted_ids, expert_ids, [used]]


def verify(outputs, expected):
    assert [x.cpu().tolist() for x in outputs] == expected


cases = [(9,257,8), (288,257,8), (1152,257,16), (2304,257,16), (33,32,32)]
with Path(sys.argv[2]).open('x') as log:
    count = 0
    for p,e,bs in cases:
        for pattern in ('spread', 'skew', 'invalid'):
            if pattern == 'spread': values = [(i*37 + i//7) % e for i in range(p)]
            elif pattern == 'skew': values = [e-1 if i%5 else 0 for i in range(p)]
            else: values = [-1 if i%3==0 else e if i%3==1 else 0 for i in range(p)]
            ids = torch.tensor(values,device='cuda',dtype=torch.int32)
            expected = reference(values,e,bs)
            for _ in range(20): verify(align(ids,e,bs),expected)
            torch.cuda.synchronize()
            for attempt in range(3):
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph,capture_error_mode='thread_local'):
                        outputs = align(ids,e,bs)
                    break
                except RuntimeError:
                    if attempt == 2: raise
                    torch.cuda.synchronize(); time.sleep(.1)
            for replay_values in (values, list(reversed(values)), values):
                ids.copy_(torch.tensor(replay_values,device='cuda',dtype=torch.int32))
                for out in outputs: out.fill_(-12345)
                graph.replay()
                verify(outputs, reference(replay_values,e,bs))
            row={'pairs':p,'experts':e,'block_size':bs,'pattern':pattern,
                 'exact':True,'repeat_launches':20,'graph_replays':3,'changed_inputs':True}
            line=json.dumps(row);print(line,flush=True);log.write(line+'\n');log.flush();count+=1
    log.write(json.dumps({'complete':True,'cases':count})+'\n')
