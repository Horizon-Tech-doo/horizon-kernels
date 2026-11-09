#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Compare single-pass checkpoint states with exact split-prefix states."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace


def run(output):
    import torch
    from horizon_kernels import gdn_ckpt
    from horizon_kernels.gdn_prefill_state import patch_module
    from vllm.third_party.flash_linear_attention.ops import chunk
    patch_module(chunk, storage_dtype='float16')
    results = []
    for K, V, H, Hg, lengths, cuts in [
        (32, 32, 2, 1, [389], [[64, 320]]),
        (64, 64, 4, 2, [1000, 313, 711], [[832, 896], [256], [640]]),
        (128, 128, 4, 2, [1000, 313, 711], [[832, 896], [], [640]]),
        (256, 64, 2, 2, [389, 771], [[320], [64, 704]]),
        (128, 128, 48, 16, [1000], [[832, 896]]),
        (128, 128, 48, 16, [4313, 7107], [[832, 4224], [7040]]),
    ]:
        torch.manual_seed(K+sum(lengths))
        qsl = [0]
        for n in lengths: qsl.append(qsl[-1]+n)
        T = qsl[-1]
        q = torch.randn(1, T, Hg, K, device='cuda').bfloat16()
        k = torch.randn_like(q)
        v = torch.randn(1, T, H, V, device='cuda').bfloat16()
        g = (-torch.rand(1, T, H, device='cuda')*.02).float()
        beta = torch.rand(1, T, H, device='cuda').bfloat16()
        h0 = (torch.randn(len(lengths), H, V, K, device='cuda')*.01).half()
        tensors = (q,k,v,g,beta)
        def op(values, initial, cu):
            return chunk.chunk_gated_delta_rule(*values, initial_state=initial,
                output_final_state=True, cu_seqlens=torch.tensor(cu,device='cuda',dtype=torch.int32),
                use_qk_l2norm_in_kernel=True)
        reference, reference_final = op(tensors, h0, qsl)
        exports = [(i,j) for i,c in enumerate(cuts) for j in range(len(c))]
        ck = SimpleNamespace(cuts=cuts, exports=exports, prefill_qsl_cpu=qsl)
        gdn_ckpt._CUR.update(ck=ck)
        try:
            actual, actual_final = op(tensors, h0, qsl)
            states = gdn_ckpt._CUR['states']
        finally:
            gdn_ckpt._CUR.clear()
        comparisons = {'output_exact': torch.equal(actual,reference),
                       'final_exact': torch.equal(actual_final,reference_final)}
        for slot,(seq,cut) in enumerate(exports):
            end = qsl[seq]+cuts[seq][cut]
            _, final = op(tuple(t[:,qsl[seq]:end].contiguous() for t in tensors),
                          h0[seq:seq+1], [0,end-qsl[seq]])
            comparisons[f'checkpoint_{slot}_exact'] = torch.equal(states[slot],final[0])
        row = dict(K=K,V=V,H=H,Hg=Hg,lengths=lengths,cuts=cuts,comparisons=comparisons)
        results.append(row); print(json.dumps(row),flush=True)
        output.write_text(json.dumps(results,indent=2)+'\n')
    assert all(all(r['comparisons'].values()) for r in results), 'checkpoint export differs'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output',type=Path)
    run(p.parse_args().output)
