#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Full FLA prefill split/cache probe using synthetic inputs, under GPU lock only.

Includes variable-length kernels, partial tails, Q/K normalization, production
head geometry and the normal autotuner. Reports the first differing intermediate;
never records prompts, traffic or model weights.
"""
import argparse
import json
import math
from pathlib import Path


def cases():
    # key/value geometry, value/key heads, prompt length and aligned cache cut.
    return [(32, 32, 2, 1, 389, 320), (64, 64, 4, 2, 389, 320),
            (128, 128, 4, 2, 389, 320), (256, 64, 2, 2, 389, 320),
            (128, 128, 48, 16, 4313, 4224),
            (128, 128, 48, 16, 7107, 7040)]


def boundaries(length, cut):
    if not 0 < cut < length or cut % 64:
        raise ValueError('cache cut must be interior and 64-token aligned')
    return [0, cut], [0, length - cut]


def run(out):
    import torch
    from vllm.triton_utils import triton
    from vllm.third_party.flash_linear_attention.ops import chunk
    from horizon_kernels.gdn_prefill_state import patch_module
    import sys

    patch_module(chunk, storage_dtype='float16')
    private = sys.modules['vllm.third_party.flash_linear_attention.ops.horizon_state_round_h']
    tuner = private.chunk_gated_delta_rule_fwd_kernel_h_blockdim64
    while not hasattr(tuner, 'configs'):
        tuner = tuner.fn
    original_configs = list(tuner.configs)
    # Diagnostic retention changes only the Python return, not GPU calculations.
    chunk.SUPPRESS_LEVEL = 3
    rows = []
    capture = {}
    original_wu = chunk.recompute_w_u_fwd

    def retain_wu(*args, **kwargs):
        w, u = original_wu(*args, **kwargs)
        capture['u'] = u
        return w, u

    chunk.recompute_w_u_fwd = retain_wu

    def measure(fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return result, capture['u']

    def diff(a, b):
        return {'exact': torch.equal(a, b),
                'max_abs': (a.float() - b.float()).abs().max().item(),
                'different_elements': (a != b).sum().item()}

    for K, V, H, Hg, T, cut in cases():
        boundaries(T, cut)
        torch.manual_seed(T + K)
        q = torch.randn(1, T, Hg, K, device='cuda').bfloat16()
        k = torch.randn_like(q)
        v = torch.randn(1, T, H, V, device='cuda').bfloat16()
        g = (-torch.rand(1, T, H, device='cuda') * .02).float()
        beta = torch.rand(1, T, H, device='cuda').bfloat16()
        initial = (torch.randn(1, H, V, K, device='cuda') * .01).half()
        for mode in ('fixed', 'autotune'):
            tuner.configs = ([triton.Config({'BV': 32}, num_warps=2, num_stages=2)]
                             if mode == 'fixed' else original_configs)
            tuner.cache = {}
            for varlen in (False, True):
                def operation(a, b, h0):
                    cu = (torch.tensor([0, b-a], device='cuda', dtype=torch.int32)
                          if varlen else None)
                    qs = chunk.l2norm_fwd(q[:, a:b].contiguous())
                    ks = chunk.l2norm_fwd(k[:, a:b].contiguous())
                    result, u = measure(chunk.chunk_gated_delta_rule_fwd,
                        q=qs, k=ks, v=v[:, a:b].contiguous(), g=g[:, a:b].contiguous(),
                        beta=beta[:, a:b].contiguous(), scale=1/math.sqrt(K),
                        initial_state=h0, output_final_state=True, cu_seqlens=cu)
                    return {'q': qs, 'k': ks, 'g': result[0], 'output': result[1],
                            'A': result[2], 'final': result[3], 'w': result[4],
                            'h': result[5], 'v_new': result[6], 'u': u}

                full = operation(0, T, initial)
                prefix = operation(0, cut, initial)
                # Compare stored fp16 resume against the checkpoint's in-call fp32 carry.
                for dtype in ('float32', 'float16'):
                    tail = operation(cut, T, prefix['final'].to(getattr(torch, dtype)))
                    compared = {}
                    for name in ('q', 'k', 'g', 'A', 'w', 'u', 'h', 'v_new', 'output', 'final'):
                        expected = full[name]
                        if name != 'final':
                            expected = expected[:, cut // 64:] if name == 'h' else expected[:, cut:]
                        compared[name] = diff(expected, tail[name])
                    first = next((n for n, d in compared.items() if not d['exact']), None)
                    row = {'K': K, 'V': V, 'H': H, 'Hg': Hg, 'T': T, 'cut': cut,
                           'varlen': varlen, 'tuner': mode, 'resume_dtype': dtype,
                           'first_difference': first, 'comparisons': compared}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    Path(out).write_text(json.dumps(rows, indent=2) + '\n')
    torch.cuda.synchronize()
    assert all(r['first_difference'] is None for r in rows), 'full prefill pipeline differs'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
