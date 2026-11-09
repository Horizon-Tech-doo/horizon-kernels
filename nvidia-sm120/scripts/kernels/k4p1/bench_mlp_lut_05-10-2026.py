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
from tilelang.contrib.horizon_k4.mlp_static_plan import StaticMlp
from tilelang.contrib.horizon_k4.mlp_mixed import MixedMlp
from tilelang.contrib.horizon_k4.mlp_mixed_lut import LutMixedMlp
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
import ctypes
from tilelang.contrib.horizon_k4.runtime import _cudart
cudart=_cudart();cudart.cudaDeviceGetAttribute.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.c_int]
regcap=ctypes.c_int();assert cudart.cudaDeviceGetAttribute(ctypes.byref(regcap),82,0)==0
assert regcap.value>0
compile_records=json.loads((root/'offline/result.json').read_text())['records']
registers={(r['rows'],r['block_n'],r['stages'],r['grid'],r['lookup']):max(map(int,r['registers'])) for r in compile_records if r['kind']=='mlp'}

emit(dict(test='hardware', name=props.name, sms=props.multi_processor_count,
          torch_version=torch.__version__))
import importlib.util
spec=importlib.util.spec_from_file_location('k4_lut_check',root/'check_silu_factor_table_05-10-2026.py')
unit=importlib.util.module_from_spec(spec);spec.loader.exec_module(unit)
unit.check_table(emit)
separator = torch.empty(256 * 1024**2, device='cuda', dtype=torch.uint8)
all_exact = True
timed_primary = 0
for hidden, intermediate, row_counts in ((5120, 17408, (4,32)),):
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
        configs = [(stages,376,128,64,lookup) for stages in (3,4) for lookup in (False,True)]
        candidates = {}
        graphs = {}
        base_graph, base_output = capture(baseline)
        base_graph.replay(); torch.cuda.synchronize()
        expected_graph = base_output.clone()
        graphs['baseline'] = base_graph
        def reference_stages():
            value = x
            stages_out = []
            for gate, down in layers:
                gu = xl.linear(value, gate)
                a = activation(gu)
                value = xl.linear(a, down)
                stages_out.append((gu, a, value))
            return stages_out
        reference_stages(); torch.cuda.synchronize()
        quality_graph, quality_outputs = capture(reference_stages)
        for stages, grid, block_n, block_k, lookup in configs:
            gu_row_warps=1
            hw, fused, policy, specialize = 1, 0, 'operation', True
            occupancy = -(-grid // props.multi_processor_count)
            name = f'gu{block_n}-down128-k{block_k}-sg{stages}-sd4-up1-static1-hw1-stages{stages}-grid{grid}-lut{int(lookup)}-operation'
            wm = 16 if rows <= 16 else 32
            shared_bound = max(stages,4) * wm * block_k * 2 + max(stages*(block_n//32),4*4)*(block_k//32)*576 + 1024
            shared_limit = getattr(props, 'shared_memory_per_multiprocessor', 0)
            if not shared_limit or occupancy * shared_bound > shared_limit:
                emit(dict(test='unsupported', hidden=hidden, rows=rows, name=name,
                          reason='co-resident shared-memory bound', bound=shared_bound, limit=shared_limit))
                continue
            regs=registers[(rows,block_n,stages,grid,lookup)]
            # Round each thread's register allocation conservatively to 16.
            reg_bound=((regs+15)//16)*16*max(block_n*gu_row_warps,128)*occupancy
            if reg_bound>regcap.value:
                emit(dict(test='unsupported',hidden=hidden,rows=rows,name=name,
                          reason='co-resident register bound',registers=regs,bound=reg_bound,limit=regcap.value))
                continue
            factory=LutMixedMlp if lookup else lambda *a,**kw:MixedMlp(*a,**kw,activation_pair=True)
            blocks = [factory(rows, hidden, intermediate, (sg, sd), grid,
                      stages=stages, block_k=block_k, block_n=block_n, stages_down=4, block_n_down=128) for _ in layers]
            emit(dict(test='candidate', hidden=hidden, rows=rows, name=name, splits=[sg, sd],
                      schedule=blocks[0].schedule.stats(), shared_memory_bound=shared_bound))
            candidate_layers = layers
            exact = True
            for scale in (1.0, 256.0):
                x.normal_(); x.mul_(scale)
                reference = x
                value = x
                for block, (gate, down), (cg, cd) in zip(blocks, layers, candidate_layers):
                    gu = xl.linear(reference, gate)
                    a = activation(gu)
                    reference = xl.linear(a, down)
                    value = block(value, cg, cd)
                    torch.cuda.synchronize(); block.check()
                    exact &= all(torch.isfinite(t).all().item() for t in (gu, a, reference, value))
                    exact &= all(torch.equal(t, r) for t, r in
                                 ((block.gu, gu), (block.a, a), (value, reference)))
                    for stage, tensor, ref in (('GU',block.gu,gu),('ACT',block.a,a),('DOWN',value,reference)):
                        if not torch.equal(tensor,ref):
                            finite=torch.isfinite(tensor).all().item() and torch.isfinite(ref).all().item()
                            emit(dict(test='numerical_mismatch', hidden=hidden,rows=rows,name=name,scale=scale,
                                      stage=stage,finite=finite,mismatch_elements=int(torch.count_nonzero(tensor!=ref)),
                                      max_abs=float((tensor.float()-ref.float()).abs().max()) if finite else None))
            emit(dict(test='eager_correctness', hidden=hidden, rows=rows, name=name, exact=exact))
            all_exact &= exact
            if not exact:
                continue
            def chain(blocks=blocks, candidate_layers=candidate_layers):
                value = x
                for block, (gate, down) in zip(blocks, candidate_layers):
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
            # Inputs mutate at fixed addresses; both graphs must consume the new
            # values, including intermediate handoffs, without resetting counters.
            replay_scales = (1.0, 256.0)
            for replay in range(64):
                x.normal_(); x.mul_(replay_scales[replay % 2])
                quality_graph.replay(); graph.replay(); torch.cuda.synchronize()
                for layer_index, (block, refs) in enumerate(zip(blocks, quality_outputs)):
                    block.check()
                    for stage, tensor, ref in zip(('GU','ACT','DOWN'), (block.gu,block.a,block.y), refs):
                        assert torch.isfinite(tensor).all().item() and torch.isfinite(ref).all().item(), (rows,name,replay,stage,'nonfinite')
                        assert torch.equal(tensor, ref), (rows,name,replay,layer_index,stage,'changed-input identity')
                assert torch.equal(result, quality_outputs[-1][-1])
            emit(dict(test='changed_input_graph_correctness', hidden=hidden, rows=rows,
                      name=name, exact=True, counters_reset=True, replays=64,
                      scales=list(replay_scales), layers=4, stages=['GU','ACT','DOWN']))
            candidates[name] = blocks
            graphs[name] = graph
        if hidden == 5120 and rows == 32:
            bad = LutMixedMlp(rows, hidden, intermediate, (sg, sd), props.multi_processor_count, stages=3,block_n=128)
            bad.ts['counters'][:sd].fill_(-2 * intermediate // (128 * sd))
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
