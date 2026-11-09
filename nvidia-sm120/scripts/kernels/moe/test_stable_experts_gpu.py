# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Stable GPU alignment through real Marlin expert projections; no speed claim.

Args: frozen candidate directory, output JSONL. Three real layers, M32/128/256.
Compare gate/up, down and final outputs to CPU-canonical alignment. Repeat fresh
GPU alignment 20 times, then replay a graph with original/reversed/original inputs.
"""
import json
from pathlib import Path
import sys
import time
root, output = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(root))
import moe_det_isolate as base
from stable_align import align

t, torch, F, mm = base.t, base.torch, base.F, base.mm
original = mm.ops.moe_wna16_marlin_gemm
snapshots = []
def traced(*args, **kwargs):
    out = original(*args, **kwargs)
    snapshots.append(out.clone())
    return out


def evaluate(layer, x, weights, ids, alignment):
    snapshots.clear()
    bs = t.hmoe._marlin_block_size(x.shape[0], t.TOPK + 1, t.E + 1)
    t.hmoe._ALIGN[0] = (ids, bs, t.E + 1, alignment)
    before = t.hmoe._ALIGN_STATS['reused']
    y = t.moe(layer.m257, x, weights, ids)
    assert t.hmoe._ALIGN_STATS['reused'] == before + 1
    assert len(snapshots) == 2
    return (*snapshots, y)


def equal(actual, expected):
    assert len(actual) == len(expected) == 3
    for stage, (a, b) in enumerate(zip(actual, expected)):
        assert torch.isfinite(a).all(), ('nonfinite', stage)
        assert torch.equal(a, b), ('canonical mismatch', stage)


def main():
    mm.ops.moe_wna16_marlin_gemm = traced
    count = 0
    try:
        with base.set_current_vllm_config(base.VllmConfig()), output.open('x') as log:
            for layer_id in (3, 7, 11):
                layer = t.Layer(layer_id)
                for m in (32, 128, 256):
                    torch.manual_seed(501 + layer_id + m)
                    x = (torch.randn(m, t.H, device='cuda') * .5).to(torch.bfloat16)
                    weights, ids = t.router_torch(x, F.linear(x, layer.gate264))
                    bs = t.hmoe._marlin_block_size(m, t.TOPK+1, t.E+1)
                    canonical = base.canonical_alignment(ids, bs, t.E+1)
                    expected = tuple(v.clone() for v in evaluate(layer, x, weights, ids, canonical))
                    def run():
                        alignment = align(ids, t.E+1, bs)
                        return evaluate(layer, x, weights, ids, alignment)
                    for _ in range(20):
                        candidate = align(ids, t.E+1, bs)
                        assert all(torch.equal(a,b) for a,b in zip(candidate,canonical))
                        equal(evaluate(layer,x,weights,ids,candidate), expected)
                    originals = [v.clone() for v in (x, weights, ids)]
                    run(); torch.cuda.synchronize()
                    for attempt in range(3):
                        graph = torch.cuda.CUDAGraph()
                        try:
                            with torch.cuda.graph(graph,capture_error_mode='thread_local'):
                                captured = run()
                            break
                        except RuntimeError:
                            if attempt == 2: raise
                            torch.cuda.synchronize(); time.sleep(.1)
                    for reverse in (False, True, False):
                        for dest, src in zip((x,weights,ids),originals):
                            dest.copy_(src.flip(0) if reverse else src)
                        canonical = base.canonical_alignment(ids,bs,t.E+1)
                        expected = tuple(v.clone() for v in evaluate(layer,x,weights,ids,canonical))
                        for value in captured: value.fill_(float('nan'))
                        graph.replay(); torch.cuda.synchronize()
                        equal(captured, expected)
                    row=dict(layer=layer_id,m=m,block_size=bs,canonical_exact=True,
                             stages=['gate_up','down','final'],fresh_alignment_repeats=20,
                             graph_replays=3,changed_inputs=True,deployment_eligible=False)
                    line=json.dumps(row); print(line,flush=True); log.write(line+'\n'); log.flush()
                    count+=1
                    del graph, captured, expected, originals
                del layer
            log.write(json.dumps(dict(complete=True,cases=count,deployment_eligible=False))+'\n')
    finally:
        mm.ops.moe_wna16_marlin_gemm = original
        t.hmoe._ALIGN[0] = None


if __name__ == '__main__': main()
