# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Portable static task plans for a gated NVFP4 MLP; no CUDA imports.

Shape/precision eligibility belongs to the operation, independent of model class.
Each plan preserves GEMM split factors and each CTA follows a topological order.
"""
GU, SILU, DOWN = 1, 2, 3


def validate(rows, hidden, intermediate, splits, ncta, policy):
    if any(type(x) is not int or x <= 0 for x in (rows, hidden, intermediate, ncta, *splits)):
        raise ValueError('positive integer dimensions, splits and CTA count required')
    if rows > 32 or hidden % 128 or intermediate % 128 or len(splits) != 2:
        raise ValueError('supports 1..32 rows and 128-aligned hidden/intermediate widths')
    sg, sd = splits
    if hidden % (sg * 64) or intermediate % (sd * 128):
        raise ValueError('split ranges must contain whole GEMM/activation tiles')
    if policy not in ('operation', 'paired', 'slice'):
        raise ValueError('unknown MLP schedule')


def build(schedule_type, rows, hidden, intermediate, splits, ncta, policy='operation'):
    validate(rows, hidden, intermediate, splits, ncta, policy)
    sg, sd = splits
    sc = schedule_type(ncta)
    cg, ca = sc.counters(sd), sc.counters(sd)
    tiles = intermediate // 128
    per = tiles // sd
    cols = intermediate // sd

    def gate_tasks(slice_index):
        gate = range(slice_index * per, (slice_index + 1) * per)
        if policy == 'paired':
            order = [bx for g in gate for bx in (g, g + tiles)]
        else:
            order = list(gate) + [g + tiles for g in gate]
        return [(bx, bk, 0, [], cg + slice_index) for bx in order for bk in range(sg)]

    def activate(slice_index):
        return [(row, slice_index * cols, 0, [(cg + slice_index, 2 * per)], ca + slice_index)
                for row in range(rows)]

    def down(slice_index):
        return [(bx, slice_index, 0, [(ca + slice_index, rows)], -1) for bx in range(hidden // 128)]

    if policy == 'slice':
        for s in range(sd):
            sc.add(GU, gate_tasks(s), signals_per_task=1 / sg)
            sc.add(SILU, activate(s))
            sc.add(DOWN, down(s))
    else:
        if policy == 'operation':
            tasks = [(bx, bk, 0, [], cg + (bx % tiles) // per)
                     for bx in range(2 * tiles) for bk in range(sg)]
        else:
            tasks = [task for s in range(sd) for task in gate_tasks(s)]
        sc.add(GU, tasks, signals_per_task=1 / sg)
        sc.add(SILU, [task for s in range(sd) for task in activate(s)])
        sc.add(DOWN, [(bx, s, 0, [(ca + s, rows)], -1)
                      for bx in range(hidden // 128) for s in range(sd)])
    sc.check()
    return sc


def build_fused(schedule_type, rows, hidden, intermediate, splits, ncta, policy='operation'):
    """Each single-split producer completes gate, up and SiLU for one 128-column tile."""
    validate(rows, hidden, intermediate, splits, ncta, policy)
    sg, sd = splits
    if sg != 1:
        raise ValueError('paired activation currently requires unsplit gate/up')
    sc = schedule_type(ncta)
    ready = sc.counters(sd)
    per = intermediate // (128 * sd)
    def producer(s):
        return [(bx, 0, 0, [], ready + s) for bx in range(s*per, (s+1)*per)]
    def consumer(s):
        return [(bx, s, 0, [(ready+s, per)], -1) for bx in range(hidden // 128)]
    if policy == 'slice':
        for s in range(sd):
            sc.add(GU, producer(s))
            sc.add(DOWN, consumer(s))
    else:
        sc.add(GU, [task for s in range(sd) for task in producer(s)])
        sc.add(DOWN, [(bx, s, 0, [(ready+s, per)], -1)
                      for bx in range(hidden // 128) for s in range(sd)])
    sc.check()
    return sc


def paired_block_order(intermediate):
    """Permutation of 32-column blocks into adjacent gate128/up128 pairs."""
    if type(intermediate) is not int or intermediate <= 0 or intermediate % 128:
        raise ValueError('128-aligned intermediate required')
    blocks = intermediate // 32
    return [tile*4 + part*blocks + n for tile in range(intermediate//128)
            for part in (0,1) for n in range(4)]


def build_up_fused(schedule_type, rows, hidden, intermediate, splits, ncta, policy='operation'):
    """Each up tile computes SiLU after its gate tile is ready, then publishes A.

    Gate counters are per128 columns; activation-ready counters are per DOWN
    K slice. GU waits on its gate after its own GEMM, so its schedule metadata
    describes the dependency before activation (weights/own GEMM can run ahead).
    """
    validate(rows, hidden, intermediate, splits, ncta, policy)
    sg, sd = splits
    if sg != 1 or policy != 'operation':
        raise ValueError('up-owned activation supports operation/single-split GU')
    sc=schedule_type(ncta)
    tiles=intermediate//128;per=tiles//sd
    gate=sc.counters(tiles);ready=sc.counters(sd)
    sc.add(GU,[(bx,0,0,[],gate+bx) for bx in range(tiles)] +
           [(tiles+bx,0,0,[(gate+bx,1)],ready+bx//per) for bx in range(tiles)])
    sc.add(DOWN,[(bx,bk,0,[(ready+bk,per)],-1)
                for bx in range(hidden//128) for bk in range(sd)])
    sc.check()
    return sc
