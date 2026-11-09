# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Check all MLP tasks, split completion and progress under adversarial CTA orders."""
import ast
from collections import Counter, defaultdict
import importlib.util
from pathlib import Path
import random
import sys

root = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('mlp_schedule', root/'mlp_schedule.py')
mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mlp)
node = next(n for n in ast.parse((root/'runtime.py').read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == 'Schedule')
namespace = {'NCTA_DEFAULT': 188}
exec(compile(ast.Module(body=[node], type_ignores=[]), 'runtime.Schedule', 'exec'), namespace)
Schedule = namespace['Schedule']


def simulate(sc, sg, order):
    cursor = [0] * sc.ncta
    counters = Counter()
    partial = defaultdict(set)
    completed = 0
    total = sum(map(len, sc.lists))
    while completed < total:
        before = completed
        for cta in order:
            if cursor[cta] == len(sc.lists[cta]):
                continue
            task = sc.lists[cta][cursor[cta]]
            if any(c >= 0 and counters[c] < target for c, target in ((task[4], task[5]), (task[6], task[7]))):
                continue
            signal = True
            if task[0] == mlp.GU:
                partial[task[1]].add(task[2])
                signal = len(partial[task[1]]) == sg
            if signal and task[8] >= 0:
                counters[task[8]] += 1
            cursor[cta] += 1
            completed += 1
        if completed == before:
            raise RuntimeError('MLP plan deadlocked')
    assert all(abs(counters[c] - count) < 1e-6 for c, count in sc.targets.items())


cases = 0
for hidden, intermediate, splits in ((5120, 17408, (1, 8)), (4096, 14336, (2, 8))):
    for rows in (1, 4, 16, 32):
        for ncta in (1, 32, 188, 376):
            reference = None
            for policy in ('operation', 'paired', 'slice'):
                sc = mlp.build(Schedule, rows, hidden, intermediate, splits, ncta, policy)
                tasks = Counter(task for queue in sc.lists for task in queue)
                if reference is None:
                    reference = tasks
                assert tasks == reference, 'task or dependency conservation failed'
                expected = 2 * intermediate // 128 * splits[0] + rows * splits[1] + hidden // 128 * splits[1]
                assert sum(tasks.values()) == expected
                shuffled = list(range(ncta)); random.Random(1042026).shuffle(shuffled)
                for order in (range(ncta), range(ncta - 1, -1, -1), shuffled):
                    simulate(sc, splits[0], order)
                cases += 1
bad = [(0, 5120, 17408, (1, 8), 188, 'operation'),
       (33, 5120, 17408, (1, 8), 188, 'operation'),
       (4, 5100, 17408, (1, 8), 188, 'operation'),
       (4, 5120, 17408, (3, 8), 188, 'operation'),
       (4, 5120, 17408, (1, 7), 188, 'operation'),
       (4, 5120, 17408, (1, 8), 0, 'operation'),
       (4, 5120, 17408, (1, 8), 188, 'unknown')]
for args in bad:
    try:
        mlp.validate(*args)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid operation contract accepted')
broken = mlp.build(Schedule, 4, 5120, 17408, (1, 8), 188)
for cta in range(broken.ncta):
    broken.lists[cta] = [task for task in broken.lists[cta] if task[0] != mlp.GU]
try:
    simulate(broken, 1, range(188))
except RuntimeError:
    pass
else:
    raise AssertionError('missing producer was not detected')
print(f'MLP_SCHEDULE_CPU_PASS cases={cases} invalid_controls={len(bad)} missing_producer_rejected=True')

fused_cases = 0
for rows in (1, 4, 16, 32):
    for ncta in (1, 32, 188, 376):
        reference = None
        for policy in ('operation', 'paired', 'slice'):
            sc = mlp.build_fused(Schedule, rows, 5120, 17408, (1, 8), ncta, policy)
            tasks = Counter(task for queue in sc.lists for task in queue)
            if reference is None:
                reference = tasks
            assert tasks == reference
            assert sum(tasks.values()) == 17408//128 + 5120//128*8
            shuffled = list(range(ncta)); random.Random(1042026).shuffle(shuffled)
            for order in (range(ncta), range(ncta-1,-1,-1), shuffled):
                simulate(sc, 1, order)
            fused_cases += 1
try:
    mlp.build_fused(Schedule, 4, 4096, 14336, (2, 8), 188)
except ValueError:
    pass
else:
    raise AssertionError('split gate/up unsupported fused plan accepted')
print(f'MLP_PAIRED_SCHEDULE_CPU_PASS fused_cases={fused_cases} split_guard=True')


def simulate_reset(sc, rows, hidden, sd, fused, order):
    cursor = [0] * sc.ncta
    counters, consumers = Counter(), Counter()
    total = sum(map(len, sc.lists)); completed = 0
    while completed < total:
        before = completed
        for cta in order:
            if cursor[cta] == len(sc.lists[cta]):continue
            task = sc.lists[cta][cursor[cta]]
            if any(c >= 0 and counters[c] < target for c,target in ((task[4],task[5]),(task[6],task[7]))):continue
            if task[8] >= 0:counters[task[8]] += 1
            if task[0] == mlp.SILU:
                slot = sd + task[4]; consumers[slot] += 1
                if consumers[slot] == rows:
                    counters[task[4]] = 0; consumers[slot] = 0
            if task[0] == mlp.DOWN:
                slot = task[2]; consumers[slot] += 1
                if consumers[slot] == hidden // 128:
                    counters[task[4]] = 0; consumers[slot] = 0
            cursor[cta] += 1; completed += 1
        if before == completed:raise RuntimeError('slice-reset deadlock')
    assert not any(counters.values()) and not any(consumers.values())

reset_cases = 0
for rows in (1,4,8,16,32):
    for ncta in (1,32,188,376):
        for fused in (False,True):
            sc=(mlp.build_fused if fused else mlp.build)(Schedule,rows,5120,17408,(1,8),ncta)
            shuffled=list(range(ncta));random.Random(1042026).shuffle(shuffled)
            for replay in range(3):
                for order in (range(ncta),range(ncta-1,-1,-1),shuffled):
                    simulate_reset(sc,rows,5120,8,fused,order)
            reset_cases += 1
print(f'MLP_SLICE_RESET_CPU_PASS cases={reset_cases} three_replays=True')
