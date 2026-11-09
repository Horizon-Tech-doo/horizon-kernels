#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Isolate actual FLA full/split fp16-carry equivalence, under the GPU lock only."""
import json
from pathlib import Path
import sys
import torch
from vllm.triton_utils import triton
from vllm.third_party.flash_linear_attention.ops import chunk
from horizon_kernels.gdn_prefill_state import patch_module


def pin_tuner(fn):
    while not hasattr(fn, 'configs'):
        fn = fn.fn
    fn.configs = [triton.Config({'BV': 32}, num_warps=2, num_stages=2)]
    fn.cache = {}


def main(out):
    from vllm.third_party.flash_linear_attention.ops import chunk_delta_h
    pin_tuner(chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    original = chunk.chunk_gated_delta_rule_fwd_h
    patch_module(chunk, storage_dtype='float16')
    private = sys.modules['vllm.third_party.flash_linear_attention.ops.horizon_state_round_h']
    pin_tuner(private.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
    corrected = chunk.chunk_gated_delta_rule_fwd_h
    results = []
    torch.manual_seed(13)
    for K, V, H, Hg in ((32, 32, 2, 1), (64, 64, 4, 2),
                         (128, 128, 4, 2), (256, 64, 2, 2)):
        T = 384
        k = (torch.randn(1, T, Hg, K, device='cuda') * .05).bfloat16()
        w = (torch.randn(1, T, H, K, device='cuda') * .02).bfloat16()
        u = (torch.randn(1, T, H, V, device='cuda') * .1).bfloat16()
        g = torch.full((1, T // 64, 64, H), -.001, device='cuda').cumsum(2).reshape(1, T, H)
        initial = (torch.randn(1, H, V, K, device='cuda') * .01).half()
        for label, fn in (('original', original), ('canonical', corrected)):
            full = fn(k, w, u, g=g, initial_state=initial, output_final_state=True)
            for cut in (64, 128, 320):
                prefix = fn(k[:, :cut].contiguous(), w[:, :cut].contiguous(), u[:, :cut].contiguous(),
                            g=g[:, :cut].contiguous(), initial_state=initial, output_final_state=True)
                tail = fn(k[:, cut:].contiguous(), w[:, cut:].contiguous(), u[:, cut:].contiguous(),
                          g=g[:, cut:].contiguous(), initial_state=prefix[2].half(), output_final_state=True)
                pairs = ((full[0][:, cut // 64:], tail[0]),
                         (full[1][:, cut:], tail[1]), (full[2], tail[2]))
                row = {'variant': label, 'K': K, 'V': V, 'H': H, 'Hg': Hg, 'cut': cut,
                       'exact': [torch.equal(a, b) for a, b in pairs],
                       'max_abs': [(a.float()-b.float()).abs().max().item() for a, b in pairs]}
                results.append(row)
                print(json.dumps(row), flush=True)
    Path(out).write_text(json.dumps(results, indent=2) + '\n')
    assert all(all(r['exact']) for r in results if r['variant'] == 'canonical'), 'canonical FLA split differs'


if __name__ == '__main__':
    main(sys.argv[1])
