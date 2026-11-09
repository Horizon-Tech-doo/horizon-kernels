# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Exploratory MLP race; output equality precedes timing, no serving hooks/deploy.

Rotates four distinct weight sets exceeding 512 MiB. Eviction is queued before
timing events, outside the measured graph. Retains all three timing rounds.
"""
import itertools
import json
from pathlib import Path
import statistics
import sys
import torch
import tilelang.contrib

root, output = map(Path, sys.argv[1:3])
tilelang.contrib.__path__.insert(0, str(root/'contrib'))
from tilelang.contrib.horizon_k4.mlp import Mlp
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl, decode_regfrag as rf
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer


def split(n, k, rows):
    cfg = xl.config(n, k, rows)
    return (tuple(cfg[1:]) if len(cfg) > 1 else rf.config(n, k, rows))[3] if cfg[0] == 'rf' else cfg[5]


def capture(fn):
    for attempt in range(4):
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                result = fn()
            return graph, result
        except RuntimeError:
            torch.cuda.synchronize()
            if attempt == 3:
                raise


out = output.open('x')
def emit(record):
    line = json.dumps(record)
    out.write(line+'\n'); out.flush(); print(line, flush=True)


props = torch.cuda.get_device_properties(0)
emit(dict(test='hardware', name=props.name, sms=props.multi_processor_count,
          torch_version=torch.__version__))
separator = torch.empty(256 * 1024**2, device='cuda', dtype=torch.uint8)
all_exact = True
timed_primary = 0
for hidden, intermediate, row_counts in ((5120, 17408, (1, 4, 8, 16, 32)), (4096, 14336, (4, 32))):
    layers = []
    for layer in range(4):
        prepared = []
        for j, (n, k) in enumerate(((2 * intermediate, hidden), (hidden, intermediate))):
            raw = synthetic_layer(n, k, seed=1042026+10*layer+j)
            prepared.append(xl.prepare(*raw)); del raw
        layers.append(tuple(prepared))
    storages = {}
    for pair in layers:
        for prep in pair:
            for key in ('Wg', 'Sg', 'alpha'):
                storage = getattr(prep, key).untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
    weight_bytes = sum(storages.values())
    if hidden == 5120:
        assert weight_bytes > 512 * 1024**2
    emit(dict(test='working_set', hidden=hidden, intermediate=intermediate,
              layers=4, weight_bytes=weight_bytes, separator_bytes=separator.numel()))
    # Independent reference: current Torch/Inductor's fused BF16 SiLU-and-mul,
    # rather than comparing the persistent activation body against itself.
    activation = torch.compile(lambda x: torch.nn.functional.silu(x[:, :intermediate]) * x[:, intermediate:],
                               fullgraph=True, dynamic=False)
    for rows in row_counts:
        sg, sd = split(2*intermediate, hidden, rows), split(hidden, intermediate, rows)
        torch.manual_seed(1042026+rows)
        x = torch.randn(rows, hidden, device='cuda', dtype=torch.bfloat16)
        def baseline():
            value = x
            for gate, down in layers:
                value = xl.linear(activation(xl.linear(value, gate)), down)
            return value
        baseline(); torch.cuda.synchronize()
        configs = [(0, occ, policy, 3) for occ, policy in itertools.product((1, 2), ('operation', 'slice'))] + [(0, 2, 'operation', False)]
        if hidden != 5120:
            configs = [(0, 1, 'operation', False), (1, 2, 'paired', False)]
        candidates = {}
        graphs = {}
        base_graph, base_output = capture(baseline)
        base_graph.replay(); torch.cuda.synchronize()
        expected_graph = base_output.clone()
        graphs['baseline'] = base_graph
        for hw, occupancy, policy, fused in configs:
            name = f'pair{int(fused)}-hw{hw}-cta{occupancy}-{policy}'
            # Conservative staged arena bound, including allocator alignment.
            wm = 16 if rows <= 16 else 32
            arena_tiles = 8 if fused >= 2 else 4
            stage_count = 3 if fused >= 2 else 4
            shared_bound = stage_count * (wm * 64 * 2 + arena_tiles * 2 * 128 * 4 + arena_tiles * 2 * 16 * 4) + 1024
            shared_limit = getattr(props, 'shared_memory_per_multiprocessor', 0)
            if occupancy == 2 and not shared_limit:
                emit(dict(test='unsupported', hidden=hidden, rows=rows, name=name, reason='unknown SM shared-memory limit'))
                continue
            if shared_limit and occupancy * shared_bound > shared_limit:
                emit(dict(test='unsupported', hidden=hidden, rows=rows, name=name,
                          reason='co-resident shared-memory bound', bound=shared_bound, limit=shared_limit))
                continue
            blocks = [Mlp(rows, hidden, intermediate, (sg, sd),
                      occupancy*props.multi_processor_count, policy=policy, hw=hw, stages=stage_count, fused=fused) for _ in layers]
            emit(dict(test='candidate', hidden=hidden, rows=rows, name=name, splits=[sg, sd],
                      schedule=blocks[0].schedule.stats(), shared_memory_bound=shared_bound))
            exact = True
            for scale in (1.0, 256.0):
                x.normal_(); x.mul_(scale)
                reference = x
                value = x
                for block, (gate, down) in zip(blocks, layers):
                    gu = xl.linear(reference, gate)
                    a = activation(gu)
                    reference = xl.linear(a, down)
                    value = block(value, gate, down)
                    torch.cuda.synchronize(); block.check()
                    exact &= all(torch.isfinite(t).all().item() for t in (gu, a, reference, value))
                    exact &= all(torch.equal(t, r) for t, r in
                                 ((block.gu, gu), (block.a, a), (value, reference)))
            emit(dict(test='eager_correctness', hidden=hidden, rows=rows, name=name, exact=exact))
            all_exact &= exact
            if not exact:
                continue
            def chain(blocks=blocks):
                value = x
                for block, (gate, down) in zip(blocks, layers):
                    value = block(value, gate, down)
                return value
            chain(); torch.cuda.synchronize()
            graph, result = capture(chain)
            # Refresh baseline after the deliberate input mutations.
            base_graph.replay(); torch.cuda.synchronize()
            expected_graph.copy_(base_output)
            for replay in range(2):
                graph.replay(); torch.cuda.synchronize()
                for block in blocks:
                    block.check()
                assert torch.equal(result, expected_graph), (hidden, rows, name, 'graph identity')
            emit(dict(test='graph_correctness', hidden=hidden, rows=rows, name=name,
                      exact=True, replays=2, counters_reset=True))
            candidates[name] = blocks
            graphs[name] = graph
        if hidden == 5120 and rows == 4:
            bad = Mlp(rows, hidden, intermediate, (sg, sd), props.multi_processor_count, stages=3, fused=3)
            bad.ts['sigtab'].fill_(-1)
            bad(x, *layers[0]); torch.cuda.synchronize()
            try:
                bad.check()
            except RuntimeError:
                emit(dict(test='missing_producer_negative_control', rejected=True,
                          device_error=int(bad.ts['err'].item()), mirror_error=int(bad.errh.item())))
            else:
                raise AssertionError('missing producer outputs were accepted')
            del bad
        if hidden != 5120:
            continue
        assert len(graphs) > 1, 'no eligible persistent MLP configuration'
        names = list(graphs)
        rounds = {name: [] for name in names}
        for round_index in range(3):
            samples = {name: [] for name in names}
            for iteration in range(20):
                order = names[(iteration+round_index) % len(names):] + names[:(iteration+round_index) % len(names)]
                for name in order:
                    separator.zero_()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); graphs[name].replay(); end.record(); end.synchronize()
                    samples[name].append(begin.elapsed_time(end)*1000/4)
            for name in names:
                rounds[name].append(statistics.median(samples[name]))
            emit(dict(test='timing_round', hidden=hidden, rows=rows, round=round_index,
                      samples_us_per_mlp=samples))
        ratios = {name: [p/c for p, c in zip(rounds['baseline'], values)]
                  for name, values in rounds.items() if name != 'baseline'}
        emit(dict(test='timing_summary', hidden=hidden, rows=rows, rounds=3, iterations=20,
                  median_us_per_mlp=rounds, speed_ratios=ratios))
        timed_primary += 1
        for blocks in candidates.values():
            for block in blocks:
                block.check()
        del graphs, candidates
    del layers
emit(dict(test='complete', primary_rows=timed_primary, all_eager_exact=all_exact,
          status='exploratory; requires fresh confirmation and serving gates before deployment'))
