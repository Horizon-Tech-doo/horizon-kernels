# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Real-weight BF16/FP8 output-projection race; no serving configuration changes.

Run only under gpu_lock.sh with serving stopped by the surrounding job. Example:
  python output_projection_race.py --hidden mainh_000.pt --bf16 model.safetensors \
      --fp8 lmhead_fp8_row.safetensors --output head-race.jsonl
No deployment or dispatch-table edits. All row/config results remain experimental.
"""
import argparse
import json
import math
from pathlib import Path


def configurations(default, n, k, smem_bytes):
    """Small sweep of pipeline depth and K tile, preserving the row/warp layout."""
    choices = [tuple(default)]
    for bk in (64, 128):
        for stages in (2, 3, 4):
            cfg = list(default)
            cfg[4], cfg[5], cfg[6] = bk, 1, stages
            choices.append(tuple(cfg))
    return list(dict.fromkeys(c for c in choices if k % c[4] == 0
                             and (k // c[4]) % c[5] == 0
                             and n % (c[2] * 32 * c[3]) == 0
                             and smem_bytes(c) <= 99 * 1024))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hidden', required=True, type=Path)
    p.add_argument('--bf16', required=True, type=Path)
    p.add_argument('--fp8', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--rows', default='1,4,8,16,17,32,33,64,65,128')
    p.add_argument('--rounds', type=int, default=3)
    args = p.parse_args()
    rows = sorted(set(int(m) for m in args.rows.split(',')))
    if not rows or min(rows) < 1 or max(rows) > 1024 or args.rounds < 3:
        p.error('rows must be 1..1024; use at least three timing rounds')

    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    from tilelang.contrib.sm120_fp8 import decode_w8a16 as w8
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    h = torch.load(args.hidden, map_location='cpu', weights_only=True)
    if isinstance(h, dict):
        h = h['h']
    h = h.reshape(-1, h.shape[-1])
    if h.shape[0] < max(rows):
        raise ValueError('captured hidden-state file has too few rows')
    # Spread across the capture instead of benchmarking only its first tokens.
    idx = torch.linspace(0, h.shape[0] - 1, max(rows)).long()
    h = h[idx].to(device='cuda', dtype=torch.bfloat16).contiguous()
    with safe_open(str(args.bf16), framework='pt', device='cpu') as f:
        weight = f.get_tensor('lm_head.weight').to('cuda')
    q = load_file(str(args.fp8), device='cuda')
    qw, scale = q['weight'], q['weight_scale'].float().reshape(-1)
    n, k = weight.shape
    assert weight.dtype == torch.bfloat16 and qw.shape == weight.shape
    assert qw.dtype == torch.float8_e4m3fn and scale.shape == (n,)
    assert h.shape[1] == k and bool(torch.isfinite(h).all())
    assert bool(torch.isfinite(scale).all()) and bool((scale > 0).all())
    # At least two distinct allocations, total footprint >512 MiB per format.
    copies = max(2, (512 * 1024**2) // (n * k) + 1)
    bfs = [weight] + [weight.clone() for _ in range(copies - 1)]
    qws = [qw] + [qw.clone() for _ in range(copies - 1)]
    preps = [w8.prepare(w, scale) for w in qws]
    sep = torch.zeros(1, device='cuda')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w') as out:
        def emit(row):
            line = json.dumps(row, allow_nan=False)
            print(line, flush=True)
            out.write(line + '\n'); out.flush()

        emit({'event': 'start', 'N': n, 'K': k, 'rows': rows, 'copies': copies,
              'fp8_rotated_bytes': copies * n * k, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'rounds': args.rounds})
        for m in rows:
            x = h[:m]
            # Chunk dequantization avoids a full 5 GB fp32 weight allocation.
            ref = torch.empty(m, n, device='cuda', dtype=torch.float32)
            for start in range(0, n, 8192):
                stop = min(start + 8192, n)
                ref[:, start:stop] = x.float() @ (
                    qw[start:stop].float() * scale[start:stop, None]).T
            baseline = x @ weight.T
            variants = [('bf16', None)] + [
                ('fp8_' + str(i), cfg) for i, cfg in enumerate(
                    configurations(w8.config(n, k, m), n, k, w8.smem_bytes))]
            graphs, stats = {}, {}
            for name, cfg in variants:
                fn = (lambda j: x @ bfs[j].T) if cfg is None else (
                    lambda j, c=cfg: w8.linear(x, preps[j], c))
                y = fn(0)
                det = bool(torch.isfinite(y).all()) and all(torch.equal(y, fn(0)) for _ in range(4))
                err = float(((y.float() - ref).norm() / ref.norm()).item()) if cfg else None
                top = y.argmax(-1)
                base_top = baseline.argmax(-1)
                dis = top != base_top
                gaps = (baseline.float().gather(1, base_top[:, None]) -
                        baseline.float().gather(1, top[:, None])).flatten()[dis]
                row = {'event': 'variant', 'M': m, 'name': name, 'cfg': cfg,
                       'deterministic': det, 'relative_error_to_dequantized_fp32': err,
                       'top1_vs_bf16': float((~dis).float().mean().item()),
                       'max_bf16_logit_gap': float(gaps.max().item()) if gaps.numel() else 0.0}
                if not det or (cfg and (not math.isfinite(err) or err > .0025)):
                    emit(dict(row, eligible=False)); continue
                # Warm both rotated copies, then capture including a separator.
                for j in range(copies):
                    fn(j)
                torch.cuda.synchronize()
                for attempt in range(3):
                    try:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, capture_error_mode='thread_local'):
                            outputs = []
                            for _ in range(2):
                                for j in range(copies):
                                    outputs.append(fn(j))
                                    sep.add_(1)
                        break
                    except RuntimeError:
                        torch.cuda.synchronize()
                        if attempt == 2:
                            raise
                for _ in range(2):
                    graph.replay(); torch.cuda.synchronize()
                    if not all(torch.equal(y, value) for value in outputs):
                        raise AssertionError(f'graph replay differs: M={m} {name}')
                graphs[name] = (graph, outputs)
                stats[name] = dict(row, eligible=True, microseconds=[])
            names = list(graphs)
            if 'bf16' not in graphs:
                raise AssertionError('BF16 reference failed finite/determinism checks')
            for turn in range(args.rounds):
                # Rotate order each round to reduce systematic thermal/order bias.
                order = names[turn % len(names):] + names[:turn % len(names)]
                for name in order:
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record(); graphs[name][0].replay(); end.record()
                    end.synchronize()
                    stats[name]['microseconds'].append(begin.elapsed_time(end) * 1000 / (2 * copies))
            for row in stats.values():
                row['best_us'] = min(row['microseconds'])
                emit(row)
            emit({'event': 'row_done', 'M': m, 'eligible_variants': len(stats)})
            del graphs, stats, outputs, graph, ref, baseline, y
        emit({'event': 'complete', 'deployment_eligible': False,
              'reason': 'Kernel timing only; full model quality and speed gates still required'})


if __name__ == '__main__':
    main()
